# CiteScope — Hybrid-Retrieval PDF Chatbot with Page-Level Citations

CiteScope answers natural-language questions over a large, pre-ingested corpus of PDF documents and returns answers grounded in the source material — every claim is backed by a **source filename and page number**. It is built to run entirely on CPU with an open-source stack, with only the answer-generation LLM being a configurable hosted provider.

> Ask a question → get a streamed answer with inline `[filename, page X]` citations, the exact passages that were retrieved, and the confidence scores behind them.

---

## Why it's different from a typical "chat with your PDF" demo

| Typical demo | CiteScope |
| --- | --- |
| Upload one PDF per session, ephemeral | **Pre-ingested persistent corpus** (10–100 PDFs, 200–2000 pages each) |
| Dense-only similarity search | **Hybrid retrieval**: dense (HNSW ANN) + sparse (lexical) fused with Reciprocal Rank Fusion |
| In-memory vectors, lost on restart | **Qdrant** persistent vector store, survives restarts |
| Digital text only | **Targeted OCR** for scanned pages and embedded images (Semantic Markdown Wrapping) |
| "Trust me" answers | **Page-level citations** + retrieved-passage visualization + hallucination/citation metrics |
| Single language | **Multilingual** (`BAAI/bge-m3`, 100+ languages); answers match the question's language |

---

## Architecture

```
Browser
  └─► Frontend (React + Vite, single-page chat)
        └─► Backend (FastAPI)
              ├─► Embedding: BAAI/bge-m3   (dense + sparse, one pass, CPU)
              ├─► Vector store: Qdrant      (HNSW ANN + sparse, RRF fusion)
              └─► LLM provider: Gemini      (configurable / hosted)
```

Two execution paths with very different performance budgets:

1. **Offline ingestion** (not latency-bound): walk a PDF directory → extract text (PyMuPDF) with targeted Tesseract OCR → clean / normalize / language-tag → deterministic recursive chunking with page metadata → embed (bge-m3) → idempotent upsert into Qdrant → structured JSON-Lines error log.
2. **Interactive query** (≤5 s end-to-end, ≤2 s first token streaming): validate → embed query → hybrid search fused with RRF → optional rerank → single-pass Chain-of-Thought generation with citations → stream answer + citations + retrieved chunks to the UI.

---

## Tech stack

- **Backend:** Python, FastAPI, SSE streaming
- **Embeddings:** `BAAI/bge-m3` (dense + sparse in a single forward pass, 8192-token context, 100+ languages)
- **Vector store:** Qdrant (named dense `HNSW` + sparse vectors, server-side RRF via the Query API, on-disk persistence)
- **Extraction / OCR:** PyMuPDF + Tesseract (conditional, targeted)
- **LLM:** Google Gemini (configurable provider abstraction — swap in any provider)
- **Frontend:** React 19 + Vite, single-page chat UI
- **Packaging:** Docker + docker-compose; CPU-only, zero GPU

---

## Features

- Persistent, pre-ingested corpus — answer immediately, no per-session uploads
- Hybrid dense + sparse retrieval with Reciprocal Rank Fusion
- Deterministic recursive chunking (reproducible chunks and citations)
- Targeted OCR for scanned pages and embedded images/diagrams
- Single-pass Chain-of-Thought generation grounded only in retrieved chunks
- Inline `[filename, page X]` provenance + structured citations (one per source PDF)
- Multi-turn conversation with bounded history (10 turns / 4000 tokens)
- Optional cross-encoder/lexical reranking with timeout fallback
- Evaluation service: p95 latency, Recall@k, MRR, citation accuracy, hallucination rate
- Single-page UI: streamed answers, citation list, retrieved-passage inspector, pipeline visualization
- Structured JSON-Lines ingestion error log for automated diagnosis

---

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/conversations` | Start a new conversation session |
| `POST` | `/conversations/{id}/query` | Ask a question (JSON or SSE stream) |
| `DELETE` | `/conversations/{id}` | Clear conversation history |
| `GET` | `/metrics` | Latest evaluation metrics |
| `GET` | `/healthz` | Liveness + Qdrant reachability + corpus count |

**Query body:** `{ "q": "your question", "top_k": 5, "stream": true }`

**SSE events:** `token` (answer chunks) · `sources` (citations) · `chunks` (retrieved passages) · `end`

---

## Run locally

### Prerequisites
- Docker Desktop
- Node.js 20+ (for the frontend dev server)
- A Google Gemini API key

### 1. Configure environment

Create a `.env` file in the repo root:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-gemini-key
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=corpus
CORPUS_DIR=/data/corpus
INGESTION_ERROR_LOG=/data/ingestion_errors.jsonl
SKIP_INGESTION=false
VITE_API_URL=http://localhost:7860
CORPUS_HOST_DIR=./corpus
```

### 2. Add PDFs

Drop your PDF files into `./corpus/`. They are ingested on backend startup.

### 3a. Run everything with docker compose

```bash
docker compose up --build
```

- Frontend: http://localhost:80
- Backend API: http://localhost:7860
- Qdrant dashboard: http://localhost:6333/dashboard

### 3b. Or run backend + Qdrant in Docker and the frontend with Vite

```bash
# Build backend image and create a shared network
docker build -t rag-backend ./backend
docker network create ragnet

# Qdrant
docker run -d --name qdrant --network ragnet \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Backend (.env is loaded via --env-file)
docker run -d --name backend --network ragnet \
  -p 7860:7860 \
  --env-file .env \
  -v "$(pwd)/corpus:/data/corpus:ro" \
  rag-backend

# Frontend (Vite dev server on http://localhost:5173)
cd frontend
npm install
npm run dev
```

> First backend startup downloads the `bge-m3` model (~2 GB). Watch progress with `docker logs -f backend`.

---

## Project layout

```
backend/
  app.py                 FastAPI app + conversation endpoints
  config.py              Env-based config with validation
  data_models.py         Core dataclasses
  embedding.py           bge-m3 wrapper (dense + sparse)
  vector_store.py        Qdrant adapter (HNSW + sparse, RRF)
  retrieval.py           Query validation + hybrid search
  reranker.py            Optional reranking with timeout fallback
  generation.py          Single-pass CoT answer generation + citations
  conversation.py        Bounded multi-turn history
  evaluation.py          Latency + retrieval/answer quality metrics
  providers/             LLM provider abstraction (Gemini)
  ingestion/             extract · clean · language · chunker · pipeline · run
  tests/                 Unit, property, and integration tests

frontend/
  src/App.jsx            Single-page chat
  src/components/        CitationList · RetrievedChunks · PipelineStages
  src/services/api.js    SSE streaming client

docker-compose.yml       Qdrant + backend + frontend
```

---

## Testing

```bash
cd backend
python -m pytest tests/ -v
```

The suite includes unit tests, property-based tests (Hypothesis), and integration tests covering the ingestion and query paths.

---

## License

MIT
