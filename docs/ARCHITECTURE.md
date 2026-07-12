# Architecture Overview

CiteRAG is a Retrieval-Augmented Generation (RAG) system designed for question answering over PDF documents with verifiable, page-level citations.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│  React 19 + Vite + Axios + i18next + Dark Mode              │
└──────────────────┬──────────────────────────────────────────┘
                   │ HTTP + SSE
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Ingestion   │  │  Retrieval   │  │  Generation  │      │
│  │   Pipeline   │  │   Service    │  │   Service    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                   │              │
│         ▼                 ▼                   ▼              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Embedding   │  │    Reranker  │  │ Conversation │      │
│  │  (BGE-M3)    │  │  (Optional)  │  │    Store     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │           Vector Store (Qdrant)                  │       │
│  │     Dense (HNSW) + Sparse + RRF Fusion          │       │
│  └──────────────────────────────────────────────────┘       │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  LLM Provider   │
                  │  (Google Gemini)│
                  └─────────────────┘
```

## Core Components

### 1. Ingestion Pipeline

**Path**: `backend/ingestion/`

Transforms raw PDFs into searchable, citable chunks:

```
PDF → Extract → Clean → Chunk → Embed → Store
```

#### Extraction (`extract.py`)
- PyMuPDF for text extraction
- Tesseract OCR for:
  - Zero-text pages (full-page OCR)
  - Embedded images (bounding-box OCR)
- Preserves page numbers for citations

#### Cleaning (`clean.py`)
- Whitespace normalization
- Header/footer removal
- Repeated pattern detection
- Unicode cleanup

#### Language Detection (`language.py`)
- `langdetect` for chunk-level language identification
- Uses language detection for chunk metadata and answer-language guidance

#### Chunking (`chunker.py`)
- Deterministic recursive token-based splitting
- Preserves sentence boundaries
- Maintains page metadata for citations
- Configurable chunk size (default: 800 tokens)

#### Embedding (`embedding.py`)
- **Model**: BAAI/bge-m3
- **Features**:
  - Dense vectors (1024-dim)
  - Sparse vectors (lexical)
  - Single-pass encoding
  - 8192-token context window
  - multilingual embeddings from BGE-M3

### 2. Vector Store

**Component**: `vector_store.py`

**Technology**: Qdrant

**Capabilities**:
- HNSW index for dense ANN search
- Sparse vector support for lexical matching
- Server-side Reciprocal Rank Fusion (RRF)
- Metadata filtering
- Two modes:
  - **Embedded**: In-process, disk-backed (development)
  - **Cloud**: Managed Qdrant cluster (production)

**Schema**:
```python
{
  "dense_vector": [1024],      # BGE-M3 dense embedding
  "sparse_vector": {...},       # BGE-M3 sparse embedding
  "payload": {
    "text": str,                # Chunk content
    "filename": str,            # Source filename
    "page_number": int,         # Page number (1-indexed)
    "chunk_position": int,      # Position in document
    "language": str             # Detected language
  }
}
```

### 3. Retrieval Service

**Component**: `retrieval.py`

**Pipeline**:

1. **Query Validation**
   - Length checks (10-500 chars)
   - Language detection
   - Content safety

2. **Embedding**
   - Query → BGE-M3 → dense + sparse vectors

3. **Hybrid Search**
   - Dense prefetch (top-K × 2)
   - Sparse prefetch (top-K × 2)
   - RRF fusion (combines scores)
   - Returns top-K results

4. **Optional Reranking**
   - Cross-encoder model
   - Timeout fallback to retrieval results
   - Improves precision at cost of latency

**Metrics**:
- Recall@K
- Mean Reciprocal Rank (MRR)
- p95 latency

### 4. Generation Service

**Component**: `generation.py`

**Responsibilities**:
- Passage-constrained answer generation
- Inline citation insertion
- Multi-turn conversation context
- Streaming via Server-Sent Events (SSE)

**Prompt Engineering**:
```
System: You are a helpful assistant. Answer based solely on context.
Include [filename, page X] after each claim.

Context: [retrieved chunks with page metadata]
History: [last 10 turns or 4000 tokens]
Question: [user query]

Answer:
```

**LLM Provider Abstraction**:
- Default: Google Gemini
- Pluggable via `backend/providers/`
- Supports streaming and non-streaming generation

### 5. Conversation Store

**Component**: `conversation.py`

**Features**:
- In-memory turn storage
- Bounded history (10 turns / 4000 tokens)
- Session lifecycle management
- Automatic cleanup

**Data Model**:
```python
Turn = {
  "question": str,
  "answer": str,
  "timestamp": datetime
}

Conversation = {
  "id": str,
  "history": List[Turn]
}
```

### 6. Evaluation Service

**Component**: `evaluation.py`

**Metrics**:
- **Retrieval**: Recall@K and MRR
- **Latency**: p95
- **Citations**: Accuracy when labeled source pages are supplied
- **Quality**: Hallucination rate when verified claims are supplied

## Data Flow

### Ingestion Flow

```
1. POST /upload → FastAPI
2. Load PDF → PyMuPDF
3. Extract text + OCR → Raw text with page numbers
4. Clean → Normalized text
5. Detect language → Language tags
6. Chunk → Token-bounded segments with page metadata
7. Embed → BGE-M3 dense + sparse vectors
8. Upsert → Qdrant with idempotent IDs
```

### Query Flow

```
1. POST /conversations/{id}/query → FastAPI
2. Validate query → Length, content, language
3. Embed query → BGE-M3
4. Hybrid search → Qdrant (dense + sparse + RRF)
5. Optional rerank → Cross-encoder
6. Fetch conversation history → Last 10 turns
7. Generate answer → Gemini with streaming
8. Parse citations → [filename, page X]
9. Persist turn → Conversation store
10. Stream response → SSE to frontend
```

## API Endpoints

### Conversations

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/conversations` | Create conversation session |
| `POST` | `/conversations/{id}/query` | Submit query (JSON or SSE) |
| `DELETE` | `/conversations/{id}` | Delete conversation |

### Documents

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/upload` | Upload and ingest PDF |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{filename}` | Serve document file |
| `DELETE` | `/documents/{filename}` | Delete document |

### Observability

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/metrics` | Evaluation metrics |
| `GET` | `/healthz` | Health check |

## Technology Choices

### Why BGE-M3?

- **Hybrid**: Dense + sparse embeddings from one model
- **Multilingual**: Suitable for multilingual document and query embeddings
- **Local option**: Embeddings can be computed on the host instead of sent to the LLM provider

### Why Qdrant?

- **Hybrid search**: Dense + sparse + RRF out-of-the-box
- **Features**: Filtering, payloads, and local or server-backed deployment options

### Why FastAPI?

- **Performance**: Async/await, ASGI
- **DX**: Auto docs, validation, type hints
- **Ecosystem**: Rich middleware, plugin support

### Why React 19?

- **Component model**: The UI is built from small React components
- **Ecosystem**: Mature browser and accessibility tooling
- **Tooling**: Vite for local development and builds

## Runtime and deployment notes

- No benchmark results are published with this repository. Ingestion and query latency depend on document size, OCR work, model loading, hardware, reranking, and the hosted LLM response.
- Embedded Qdrant and the in-memory conversation store are the default local-development path.
- Qdrant Cloud, a reverse proxy, authentication, rate limiting, and a shared session store are deployment options that still need to be configured and validated for a public service.

## Extension Points

1. **LLM Providers**: Add new providers in `backend/providers/`
2. **Rerankers**: Swap cross-encoder in `reranker.py`
3. **Embeddings**: Replace BGE-M3 in `embedding.py`
4. **Storage**: Replace Qdrant via `vector_store.py` interface
5. **Auth**: Add FastAPI middleware
6. **UI**: Extend React components in `frontend/src/components/`

## Security Considerations

- API keys stored in environment variables
- CORS is currently permissive for development; restrict allowed origins before public deployment
- Input validation on all endpoints
- Rate limiting recommended for public deployments
- SSRF protection for URL ingestion

See [SECURITY.md](../SECURITY.md) for comprehensive security guidelines.
