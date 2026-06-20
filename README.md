# CiteRAG

A Retrieval-Augmented Generation system for question answering over large PDF corpora. Answers are grounded exclusively in the ingested documents and include page-level source citations.

## Overview

CiteRAG ingests a corpus of PDF documents into a persistent vector store, then serves natural-language queries via a hybrid dense + sparse retrieval pipeline followed by single-pass answer generation. Every answer cites the source filename and page number of each claim.

The entire stack runs on CPU. The only external dependency is a configurable hosted LLM provider for answer generation.

## Architecture

```
Frontend (React/Vite)
    │
    ▼
Backend (FastAPI)
    ├── Embedding       BAAI/bge-m3 — dense + sparse vectors in one pass
    ├── Vector store    Qdrant       — HNSW ANN + sparse, server-side RRF
    └── LLM provider    Gemini       — configurable
```

**Ingestion pipeline** (offline, not latency-bound)

PDF → PyMuPDF text extraction → targeted Tesseract OCR (zero-text pages + embedded images) → cleaning + whitespace normalization + header/footer removal → language detection → deterministic recursive chunking with page metadata → `bge-m3` embedding → idempotent upsert to Qdrant

**Query pipeline** (interactive, ≤5 s end-to-end)

Query → validation → `bge-m3` embedding → Qdrant hybrid search (dense prefetch + sparse prefetch, RRF fusion) → optional reranking → single-pass Chain-of-Thought generation → streamed answer with inline citations

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Embeddings | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) — dense + sparse, 8 192-token context, 100+ languages |
| Vector store | [Qdrant](https://qdrant.tech) — HNSW ANN, sparse vectors, server-side RRF |
| PDF extraction | PyMuPDF |
| OCR | Tesseract |
| Tokenizer | tiktoken |
| LLM | Google Gemini (default) — provider-agnostic abstraction |
| Frontend | React 19, Vite |
| Deployment | Docker, docker compose |

## Features

- Pre-ingested persistent corpus — queries are served without per-request uploads
- Hybrid dense + sparse retrieval fused via Reciprocal Rank Fusion
- Deterministic recursive chunking (reproducible boundaries, reproducible citations)
- Targeted OCR — full-page OCR only on zero-text pages; image bounding-box OCR on digital pages
- Semantic Markdown Wrapping for image-derived text (no vision-language model required)
- Single-pass Chain-of-Thought generation, no agentic routing
- Inline `[filename, page X]` provenance and structured citations per source
- Multi-turn conversation with bounded history (10 turns / 4 000 tokens)
- Optional reranking with timeout fallback
- Evaluation service: p95 latency, Recall@k, MRR, citation accuracy, hallucination rate

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/conversations` | Create a conversation session |
| `POST` | `/conversations/{id}/query` | Submit a query (JSON or SSE stream) |
| `DELETE` | `/conversations/{id}` | Delete conversation history |
| `GET` | `/metrics` | Evaluation metrics |
| `GET` | `/healthz` | Health check |

**Request body** — `POST /conversations/{id}/query`
```json
{ "q": "string", "top_k": 5, "stream": true }
```

**SSE event types** — `token` · `sources` · `chunks` · `end` · `error`

## Getting Started

### Requirements

- Docker and Docker Compose
- Node.js 20+ (frontend dev server only)
- A `GOOGLE_API_KEY` for Gemini

### Configuration

Create `.env` at the repository root:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=<your-key>
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=corpus
CORPUS_DIR=/data/corpus
INGESTION_ERROR_LOG=/data/ingestion_errors.jsonl
SKIP_INGESTION=false
VITE_API_URL=http://localhost:7860
CORPUS_HOST_DIR=./corpus
```

Place PDF files in `./corpus/`. The backend ingests them on startup.

### Run with Docker Compose

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:80 |
| Backend API | http://localhost:7860 |
| Qdrant | http://localhost:6333 |

### Run backend + Qdrant in Docker, frontend with Vite

```bash
# Build and network
docker build -t citerag-backend ./backend
docker network create ragnet

# Qdrant
docker run -d --name qdrant --network ragnet \
  -p 6333:6333 -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Backend
docker run -d --name backend --network ragnet \
  -p 7860:7860 \
  --env-file .env \
  -v "$(pwd)/corpus:/data/corpus:ro" \
  citerag-backend

# Frontend
cd frontend && npm install && npm run dev
```

The first backend startup downloads the `bge-m3` model weights (~2 GB). Subsequent starts reuse the cache.

## Project Structure

```
backend/
├── app.py               FastAPI application
├── config.py            Environment-based configuration
├── data_models.py       Shared dataclasses
├── embedding.py         bge-m3 wrapper
├── vector_store.py      Qdrant adapter
├── retrieval.py         Query validation and hybrid search
├── reranker.py          Optional reranking
├── generation.py        Answer generation and citation building
├── conversation.py      Multi-turn session store
├── evaluation.py        Metrics service
├── providers/           LLM provider abstraction
├── ingestion/           Extraction, cleaning, chunking, pipeline, runner
└── tests/               Unit, property-based, and integration tests

frontend/
├── src/App.jsx          Single-page chat interface
├── src/components/      CitationList, RetrievedChunks, PipelineStages
└── src/services/api.js  HTTP + SSE client
```

## Testing

```bash
cd backend
python -m pytest tests/ -v
```

## License

MIT
