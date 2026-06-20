"""QA RAG application with multi-document user sessions.

Also exposes the new conversation-based RAG endpoints:
  POST   /conversations                → { conversation_id: str }
  POST   /conversations/{id}/query     → JSON or SSE stream
  DELETE /conversations/{id}           → 200
  GET    /metrics                      → metrics report
  GET    /healthz                      → { status, qdrant, corpus_count }
"""
import os
import uuid
import time
import pathlib
import datetime
import asyncio
import threading
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union

import json
import socket
import ipaddress
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Request
from fastapi.responses import RedirectResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import errors
import uvicorn
import httpx
import numpy as np

# Set Hugging Face Hub download timeout to 120 seconds to prevent ReadTimeoutErrors in Spaces
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# SentenceTransformer is imported lazily inside lifespan() to avoid a Windows
# access-violation crash caused by pyarrow being imported too early during the
# pytest collection phase (Python 3.13 + pyarrow incompatibility).
from utils.loaders import load_source
from utils.splitter import split_text
from utils.exceptions import DocumentLoaderError
from rag_session import RAGSession
from user_session import UserSession

# Load environment variables
load_dotenv()

# --- Configuration ---
SESSION_CLEANUP_INTERVAL_SECONDS = 300
SESSION_TIMEOUT_MINUTES = 15

# --- In-Memory Session Storage ---
sessions: Dict[str, UserSession] = {}
_session_lock = threading.Lock()

# --- Background Cleanup Logic ---
def _clean_sessions_once():
    now = datetime.datetime.now()
    expiration_time = datetime.timedelta(minutes=SESSION_TIMEOUT_MINUTES)

    with _session_lock:
        # Create a copy of the session IDs to avoid modifying the dictionary while iterating
        expired_ids = [
            session_id for session_id, session in sessions.items()
            if now - session.last_accessed > expiration_time
        ]
        for session_id in expired_ids:
            del sessions[session_id]
            print(f"Cleaned up expired user session: {session_id}")

async def cleanup_expired_sessions_task():
    while True:
        _clean_sessions_once()
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    from sentence_transformers import SentenceTransformer  # lazy import — avoids Windows pyarrow crash on test collection
    print("Loading embedding model...")
    app.state.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("Embedding model loaded.")

    print("Initializing HTTP client...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    app.state.http_client = httpx.AsyncClient(headers=headers)
    print("HTTP client initialized.")

    # --- New RAG services (Requirements 9.3, 12.1, 14.1, 14.5) ---
    print("Loading RAG embedding model (BAAI/bge-m3)...")
    try:
        from backend.embedding import EmbeddingModel
        from backend.vector_store import VectorStore
        from backend.retrieval import RetrievalService
        from backend.generation import AnswerGenerator
        from backend.conversation import ConversationStore
        from backend.evaluation import EvaluationService
        from backend.reranker import Reranker
        from backend.providers import get_provider

        rag_embedding_model = EmbeddingModel("BAAI/bge-m3")
        app.state.rag_embedding_model = rag_embedding_model
        print("RAG embedding model loaded.")

        print("Connecting to Qdrant...")
        vector_store = VectorStore()
        app.state.rag_vector_store = vector_store
        print("Qdrant connected.")

        app.state.rag_retrieval = RetrievalService(rag_embedding_model, vector_store)
        app.state.rag_generator = AnswerGenerator(get_provider())
        app.state.rag_conversations = ConversationStore()
        app.state.rag_evaluation = EvaluationService()
        rerank_enabled = os.getenv("RERANK_ENABLED", "false").lower() == "true"
        app.state.rag_reranker = Reranker(enabled=rerank_enabled)
        print(f"RAG services initialized (reranking={'enabled' if rerank_enabled else 'disabled'}).")
    except Exception as exc:
        # Non-fatal: old endpoints still work without new RAG services.
        print(f"WARNING: RAG services failed to initialize: {exc}")
        app.state.rag_embedding_model = None
        app.state.rag_vector_store = None
        app.state.rag_retrieval = None
        app.state.rag_generator = None
        app.state.rag_conversations = None
        app.state.rag_evaluation = None
        app.state.rag_reranker = None

    print("Starting session cleanup task...")
    asyncio.create_task(cleanup_expired_sessions_task())
    yield

    print("Closing HTTP client...")
    await app.state.http_client.aclose()
    print("Application shutdown.")

# --- App Initialization ---
app = FastAPI(
    title="DocQA",
    description="A RAG application supporting multi-document user sessions.",
    version="2.0.0",
    lifespan=lifespan
)

# --- LLM and CORS Configuration ---
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
# Initialize the modern google-genai Client
ai_client = genai.Client(api_key=GENAI_API_KEY)
MODEL_NAME = "gemini-3.5-flash"
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash"]

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Helper Functions ---
def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if not parsed.hostname:
            return False
        ip = socket.gethostbyname(parsed.hostname)
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
            return False
        return True
    except Exception:
        return False

async def generate_rag_response(query: str, context_chunks: List[str], stream: bool = False):
    """Generates a response from the LLM, supports streaming."""
    if not context_chunks:
        if stream:
            yield f"data: {json.dumps({'token': 'No relevant information found.'})}\n\n"
        else:
            yield "No relevant information found."
        return

    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer the following question based only on the provided context. "
        "If the user asks in a language other than English, respond in their language.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        current_model = MODEL_NAME
        if attempt > 0 and attempt - 1 < len(FALLBACK_MODELS):
            current_model = FALLBACK_MODELS[attempt - 1]

        try:
            if stream:
                response = await asyncio.wait_for(
                    ai_client.aio.models.generate_content_stream(
                        model=current_model,
                        contents=prompt
                    ),
                    timeout=30.0
                )
                async for chunk in response:
                    # Ensure the chunk has content before sending
                    if chunk.text:
                        yield f"data: {json.dumps({'token': chunk.text})}\n\n"
                return  # Exit generator on success
            else:
                response = await asyncio.wait_for(
                    ai_client.aio.models.generate_content(
                        model=current_model,
                        contents=prompt
                    ),
                    timeout=30.0
                )
                yield response.text.strip()
                return  # Exit generator on success
        except errors.APIError as e:
            # Check for HTTP 429 or 503 status code (high demand)
            if e.code in (429, 503):
                if attempt == max_retries - 1:
                    error_message = "Model is experiencing high demand. Please try again later."
                    if stream:
                        yield f"data: {json.dumps({'error': error_message})}\n\n"
                        return
                    else:
                        raise HTTPException(status_code=503, detail=error_message)
                else:
                    print(f"High demand hit for {current_model}. Retrying with fallback... (Attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
            else:
                error_message = f"LLM generation failed: {e.message}"
                if stream:
                    yield f"data: {json.dumps({'error': error_message})}\n\n"
                    return
                else:
                    raise HTTPException(status_code=500, detail=error_message)
        except asyncio.TimeoutError:
            error_message = "LLM generation timed out."
            if stream:
                yield f"data: {json.dumps({'error': error_message})}\n\n"
                return
            else:
                raise HTTPException(status_code=504, detail=error_message)
        except Exception as e:
            error_message = f"LLM generation failed: {e}"
            if stream:
                yield f"data: {json.dumps({'error': error_message})}\n\n"
                return
            else:
                raise HTTPException(status_code=500, detail=error_message)

# --- API Models ---
class SessionResponse(BaseModel):
    session_id: str

class IngestResponse(BaseModel):
    doc_id: str
    source: str
    num_chunks: int

class QueryPayload(BaseModel):
    q: str
    doc_ids: Optional[List[str]] = None
    stream: Optional[bool] = False

class QuerySource(BaseModel):
    text: str
    score: float
    doc_id: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[QuerySource]

class SessionStatusResponse(BaseModel):
    session_id: str
    active: bool
    remaining_minutes: Optional[float] = None
    last_accessed: str

class SessionRefreshResponse(BaseModel):
    session_id: str
    refreshed_at: str
    remaining_minutes: float

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/sessions", response_model=SessionResponse, summary="Create a new user session")
async def create_session():
    session_id = uuid.uuid4().hex
    with _session_lock:
        sessions[session_id] = UserSession()
    return SessionResponse(session_id=session_id)

@app.post("/sessions/{session_id}/ingest", response_model=IngestResponse, summary="Ingest a document into a session")
async def ingest(session_id: str, request: Request):
    with _session_lock:
        user_session = sessions.get(session_id)
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")

    file_filename = None
    file_content = None
    url = None
    has_file = False

    body_bytes = await request.body()

    # 1. Try to parse as JSON URL first
    try:
        body = json.loads(body_bytes)
        url = body.get("url")
    except Exception:
        pass

    # 2. If no URL was successfully parsed, try parsing as Form data
    if not url:
        try:
            form = await request.form()
            file = form.get("file")
            if file and hasattr(file, "filename") and file.filename:
                file_filename = file.filename
                file_content = await file.read()
                has_file = True
        except Exception:
            pass

    if not has_file and not url:
        content_type = request.headers.get("content-type", "")
        headers_str = str(dict(request.headers))
        try:
            body_preview = (await request.body())[:200]
            body_preview_str = body_preview.decode("utf-8", errors="replace")
        except Exception as e:
            body_preview_str = f"could not read body: {e}"
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provide either a file (multipart/form-data) or a URL (application/json). "
                f"Received Content-Type: {content_type}. Headers: {headers_str}. "
                f"Body preview: {body_preview_str}"
            )
        )
    if has_file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL, not both.")

    source_name = ""
    content = b""
    source_ext = "url"

    if has_file:
        source_name = file_filename
        content = file_content
        source_ext = pathlib.Path(source_name).suffix or "url"
    elif url:
        if not is_safe_url(url):
            raise HTTPException(status_code=400, detail="Invalid or restricted URL provided.")
        source_name = url
        try:
            jina_url = f"https://r.jina.ai/{url}"
            print(f"Fetching URL via Jina Reader: {jina_url}")
            response = await app.state.http_client.get(jina_url, timeout=30.0)
            response.raise_for_status()
            content = response.content
            source_ext = "md"  # Jina Reader returns Markdown content
        except Exception as e:
            print(f"Jina Reader fetch failed: {e}. Falling back to direct URL fetch...")
            try:
                response = await app.state.http_client.get(url, timeout=30.0)
                response.raise_for_status()
                content = response.content
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                source_ext = pathlib.Path(parsed_url.path).suffix or "url"
            except httpx.HTTPStatusError as e_direct:
                raise HTTPException(status_code=e_direct.response.status_code, detail=f"Failed to fetch URL: {e_direct.response.text}")
            except httpx.RequestError as e_direct:
                raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {e_direct}")

    try:
        text = load_source(content, source_ext)
    except DocumentLoaderError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the provided source.")

    chunks = split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="The document is too short to be processed.")

    # --- Caching and Embedding Logic ---
    # This logic checks the user's session cache for existing chunk embeddings.
    # It only sends chunks that have not been seen before to the embedding model,
    # avoiding redundant, expensive computations.
    ordered_embeddings = [None] * len(chunks)
    chunks_to_encode = []
    indices_of_new_chunks = []

    # Identify which chunks are new and which are cached
    for i, chunk in enumerate(chunks):
        if chunk in user_session.embedding_cache:
            ordered_embeddings[i] = user_session.embedding_cache[chunk]
        else:
            chunks_to_encode.append(chunk)
            indices_of_new_chunks.append(i)

    # If there are new chunks, encode them in a single batch for efficiency
    if chunks_to_encode:
        # Encode each unique new chunk only once to save computation
        unique_new_chunks = list(dict.fromkeys(chunks_to_encode))

        try:
            generated_embeddings = await asyncio.wait_for(
                asyncio.to_thread(
                    app.state.embedding_model.encode, unique_new_chunks, convert_to_numpy=True
                ),
                timeout=180.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Embedding generation timed out.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

        new_embeddings_dict = {
            chunk: emb for chunk, emb in zip(unique_new_chunks, generated_embeddings)
        }

        # Add the newly generated embeddings to the session cache for future use
        user_session.embedding_cache.update(new_embeddings_dict)

        # Place the new embeddings into the final ordered list
        for i, chunk in enumerate(chunks_to_encode):
            original_index = indices_of_new_chunks[i]
            ordered_embeddings[original_index] = new_embeddings_dict[chunk]

    all_embeddings_np = np.array(ordered_embeddings)
    # --- End of Caching and Embedding Logic ---

    doc_id = uuid.uuid4().hex
    rag_session = RAGSession(source=source_name, embedding_model=app.state.embedding_model)
    # Pass the pre-computed embeddings to the new ingest method
    rag_session.ingest(chunks, all_embeddings_np)
    user_session.add_doc(doc_id, rag_session)

    return IngestResponse(doc_id=doc_id, source=source_name, num_chunks=len(chunks))

@app.post("/sessions/{session_id}/query", summary="Ask a question within a session")
async def query(session_id: str, payload: QueryPayload):
    with _session_lock:
        user_session = sessions.get(session_id)
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")

    user_session.touch()

    # Determine which documents to query.
    docs_to_query_items = user_session.docs.items()
    if payload.doc_ids:
        docs_to_query_items = [
            (doc_id, user_session.get_doc(doc_id))
            for doc_id in payload.doc_ids
            if user_session.get_doc(doc_id) is not None
        ]

    all_chunks = []
    for doc_id, rag_session in docs_to_query_items:
        try:
            retrieved = await rag_session.query(payload.q, k=5)
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query search failed: {e}")

        for chunk in retrieved:
            chunk['doc_id'] = doc_id
            chunk['source'] = rag_session.source
        all_chunks.extend(retrieved)

    all_chunks.sort(key=lambda x: x['score'], reverse=True)
    top_chunks = all_chunks[:5]

    relevant_sources = [QuerySource(**chunk) for chunk in top_chunks]
    relevant_texts = [chunk['text'] for chunk in top_chunks]

    # If streaming is requested, return a StreamingResponse
    if payload.stream:
        async def stream_generator():
            # First, send an event with the sources
            sources_data = [s.model_dump() for s in relevant_sources]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

            # Then, stream the LLM response tokens
            async for chunk in generate_rag_response(payload.q, relevant_texts, stream=True):
                yield chunk

            # Signal the end of the stream
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # If not streaming, use the original logic
    else:
        answer = ""
        # The async generator yields one result in non-streaming mode
        async for content in generate_rag_response(payload.q, relevant_texts, stream=False):
            answer = content
        return QueryResponse(answer=answer, sources=relevant_sources)

@app.get("/sessions/{session_id}/status", response_model=SessionStatusResponse, summary="Get session status and remaining time")
async def get_session_status(session_id: str):
    """Returns session status, activity state, and remaining time before expiration."""
    now = datetime.datetime.now()
    expiration_time = datetime.timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    
    with _session_lock:
        user_session = sessions.get(session_id)
    
    if not user_session:
        return SessionStatusResponse(
            session_id=session_id,
            active=False,
            last_accessed=now.isoformat()
        )
    
    time_since_access = now - user_session.last_accessed
    remaining_time = expiration_time - time_since_access
    
    if remaining_time.total_seconds() <= 0:
        return SessionStatusResponse(
            session_id=session_id,
            active=False,
            last_accessed=user_session.last_accessed.isoformat()
        )
    
    return SessionStatusResponse(
        session_id=session_id,
        active=True,
        remaining_minutes=remaining_time.total_seconds() / 60,
        last_accessed=user_session.last_accessed.isoformat()
    )

@app.post("/sessions/{session_id}/refresh", response_model=SessionRefreshResponse, summary="Refresh session to extend timeout")
async def refresh_session(session_id: str):
    """Refreshes a session to extend its timeout period."""
    with _session_lock:
        user_session = sessions.get(session_id)
    
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")
    
    user_session.touch()
    
    return SessionRefreshResponse(
        session_id=session_id,
        refreshed_at=user_session.last_accessed.isoformat(),
        remaining_minutes=SESSION_TIMEOUT_MINUTES
    )

@app.get("/sessions/{session_id}/health", summary="Simple session health check")
async def session_health_check(session_id: str):
    """Simple endpoint to check if session exists and is active."""
    now = datetime.datetime.now()
    expiration_time = datetime.timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    
    with _session_lock:
        user_session = sessions.get(session_id)
    
    if not user_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    time_since_access = now - user_session.last_accessed
    if time_since_access > expiration_time:
        raise HTTPException(status_code=410, detail="Session expired")
    
    return {"status": "active"}

@app.delete("/sessions/{session_id}/documents/{doc_id}", status_code=204, summary="Delete a document from a session")
async def delete_document(session_id: str, doc_id: str):
    """Deletes a specific document from a user session."""
    with _session_lock:
        user_session = sessions.get(session_id)
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")

    if not user_session.get_doc(doc_id):
        raise HTTPException(status_code=404, detail="Document not found in this session.")

    user_session.remove_doc(doc_id)
    return

# ---------------------------------------------------------------------------
# New conversation-based RAG endpoints
# Requirements: 9.3, 12.1, 12.2, 12.3, 12.6, 14.1, 14.5
# ---------------------------------------------------------------------------


def _require_rag(request: Request) -> None:
    """Raise 503 if the RAG services failed to initialise."""
    if request.app.state.rag_retrieval is None:
        raise HTTPException(
            status_code=503,
            detail="RAG services are unavailable. Check server logs for initialisation errors.",
        )


# --- Pydantic models for new endpoints ---

class ConversationResponse(BaseModel):
    conversation_id: str


class ConversationQueryPayload(BaseModel):
    q: str
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    stream: Optional[bool] = False


class CitationOut(BaseModel):
    filename: str
    pages: List[int]


class ChunkOut(BaseModel):
    text: str
    score: float
    filename: str
    page_number: int
    chunk_position: int
    language: str


class ConversationQueryResponse(BaseModel):
    answer: str
    citations: List[CitationOut]
    chunks: List[ChunkOut]


# --- POST /conversations ---

@app.post(
    "/conversations",
    response_model=ConversationResponse,
    summary="Start a new conversation session",
    tags=["RAG Conversations"],
)
async def create_conversation(request: Request):
    """Create a new Conversation_Session and return its id.

    Requirements: 12.1
    """
    _require_rag(request)
    conversation_id = uuid.uuid4().hex
    # Register the session (empty history) so it is known to the store.
    # ConversationStore lazily creates sessions on first append, but pre-
    # registering via clear() is safe and keeps behaviour consistent.
    return ConversationResponse(conversation_id=conversation_id)


# --- POST /conversations/{id}/query ---

@app.post(
    "/conversations/{conversation_id}/query",
    summary="Submit a query within a conversation",
    tags=["RAG Conversations"],
)
async def conversation_query(
    conversation_id: str,
    payload: ConversationQueryPayload,
    request: Request,
):
    """Retrieve relevant chunks, generate an answer, and record the turn.

    Supports both JSON (`stream=false`) and SSE streaming (`stream=true`).

    SSE event types emitted when streaming:
    - ``{"type": "token",   "text": "..."}``   — incremental answer tokens
    - ``{"type": "sources", "data": [...]}``    — structured citations
    - ``{"type": "chunks",  "data": [...]}``    — retrieved passage metadata
    - ``{"type": "end"}``                       — stream finished normally
    - ``{"type": "error",   "message": "..."}`` — generation error

    Requirements: 9.3, 12.2, 12.3
    """
    _require_rag(request)

    retrieval_svc = request.app.state.rag_retrieval
    generator = request.app.state.rag_generator
    conversation_store = request.app.state.rag_conversations
    evaluation_svc = request.app.state.rag_evaluation
    reranker = request.app.state.rag_reranker

    from backend.retrieval import InvalidQueryError
    from backend.conversation import Turn

    top_k = payload.top_k if payload.top_k is not None else 5
    query = payload.q

    # 1. Fetch conversation history (Req 12.2, 12.3)
    history = conversation_store.history(conversation_id)

    # 2. Start latency timer (Req 9.3)
    t_start = time.monotonic()

    async def _run_retrieval_and_generation():
        """Inner coroutine; returns (result, candidates) or raises."""
        # Validate + retrieve (Req 6.6, 6.3)
        retrieval_result = await asyncio.to_thread(retrieval_svc.retrieve, query, top_k)
        candidates = retrieval_result.candidates

        # Optional reranking (Req 7.1–7.5)
        if reranker.enabled and candidates:
            candidates = await asyncio.to_thread(reranker.rerank, query, candidates)

        # Generate answer (Req 8.1)
        gen_result = await generator.generate(query, candidates, history)
        return gen_result, candidates

    # --- Streaming path ---
    if payload.stream:
        async def sse_generator():
            t0 = time.monotonic()
            try:
                gen_result, candidates = await _run_retrieval_and_generation()

                # Stream answer tokens
                if gen_result.error:
                    yield f"data: {json.dumps({'type': 'error', 'message': gen_result.error})}\n\n"
                    return

                # Yield the answer as tokens (word-by-word for streaming effect)
                for word in gen_result.answer.split(" "):
                    yield f"data: {json.dumps({'type': 'token', 'text': word + ' '})}\n\n"
                    await asyncio.sleep(0)  # yield to event loop

                # Persist the turn (Req 12.1, 12.4)
                conversation_store.append(
                    conversation_id,
                    Turn(question=query, answer=gen_result.answer),
                )

                # Emit citations (sources)
                sources_data = [
                    {"filename": c.filename, "pages": c.pages}
                    for c in gen_result.citations
                ]
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                # Emit retrieved chunks
                chunks_data = [
                    {
                        "text": cand.text,
                        "score": cand.score,
                        "filename": cand.metadata.filename,
                        "page_number": cand.metadata.page_number,
                        "chunk_position": cand.metadata.chunk_position,
                        "language": cand.metadata.language,
                    }
                    for cand in candidates
                ]
                yield f"data: {json.dumps({'type': 'chunks', 'data': chunks_data})}\n\n"

                # End event
                yield f"data: {json.dumps({'type': 'end'})}\n\n"

            except InvalidQueryError as exc:
                err_msg = str(exc)
                yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
            except Exception as exc:
                err_msg = f"Query failed: {exc}"
                yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
            finally:
                # Record latency for every handled query including errors (Req 9.3)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                evaluation_svc.record_latency_ms(elapsed_ms)

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- JSON (non-streaming) path ---
    try:
        gen_result, candidates = await _run_retrieval_and_generation()

        # Record latency (Req 9.3)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        evaluation_svc.record_latency_ms(elapsed_ms)

        if gen_result.error:
            raise HTTPException(status_code=500, detail=gen_result.error)

        # Persist the turn (Req 12.1, 12.4)
        conversation_store.append(
            conversation_id,
            Turn(question=query, answer=gen_result.answer),
        )

        citations_out = [
            CitationOut(filename=c.filename, pages=c.pages)
            for c in gen_result.citations
        ]
        chunks_out = [
            ChunkOut(
                text=cand.text,
                score=cand.score,
                filename=cand.metadata.filename,
                page_number=cand.metadata.page_number,
                chunk_position=cand.metadata.chunk_position,
                language=cand.metadata.language,
            )
            for cand in candidates
        ]
        return ConversationQueryResponse(
            answer=gen_result.answer,
            citations=citations_out,
            chunks=chunks_out,
        )

    except InvalidQueryError as exc:
        # Record latency even for validation errors (Req 9.3)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        evaluation_svc.record_latency_ms(elapsed_ms)
        raise HTTPException(status_code=422, detail=str(exc))
    except HTTPException:
        # Record latency even on HTTP errors (Req 9.3)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        evaluation_svc.record_latency_ms(elapsed_ms)
        raise
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        evaluation_svc.record_latency_ms(elapsed_ms)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


# --- DELETE /conversations/{id} ---

@app.delete(
    "/conversations/{conversation_id}",
    status_code=200,
    summary="Clear conversation history",
    tags=["RAG Conversations"],
)
async def delete_conversation(conversation_id: str, request: Request):
    """Clear all history for the given conversation session.

    Requirements: 12.6
    """
    _require_rag(request)
    request.app.state.rag_conversations.clear(conversation_id)
    return {"conversation_id": conversation_id, "cleared": True}


# --- GET /metrics ---

@app.get(
    "/metrics",
    summary="Latest evaluation metrics with sample sizes",
    tags=["RAG Conversations"],
)
async def get_metrics(request: Request):
    """Return the most recently computed evaluation metrics.

    Requirements: 10.7
    """
    _require_rag(request)
    report = request.app.state.rag_evaluation.report()

    def _metric_to_dict(m):
        if m is None:
            return None
        return {
            "value": m.value,
            "sample_size": m.sample_size,
            "error": m.error,
            "is_insufficient": m.is_insufficient,
        }

    return {
        "p95_latency": _metric_to_dict(report.p95_latency),
        "recall_at_k": _metric_to_dict(report.recall_at_k),
        "mrr": _metric_to_dict(report.mrr),
        "citation_accuracy": _metric_to_dict(report.citation_accuracy),
        "hallucination_rate": _metric_to_dict(report.hallucination_rate),
    }


# --- GET /healthz ---

@app.get(
    "/healthz",
    summary="Liveness check with Qdrant reachability and corpus count",
    tags=["RAG Conversations"],
)
async def healthz(request: Request):
    """Check that the API is live, Qdrant is reachable, and report corpus size.

    Requirements: 14.5
    """
    qdrant_ok = False
    corpus_count = 0

    vector_store = request.app.state.rag_vector_store
    if vector_store is not None:
        try:
            corpus_count = await asyncio.to_thread(vector_store.count)
            qdrant_ok = True
        except Exception:
            qdrant_ok = False

    status_code = 200 if qdrant_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if qdrant_ok else "degraded",
            "qdrant": qdrant_ok,
            "corpus_count": corpus_count,
        },
    )


# ... (Main Execution block, no changes)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
