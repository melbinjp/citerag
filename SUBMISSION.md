# CiteScope — Hackathon Submission

**Challenge 1: RAG Chatbot** — answer user queries from a large private corpus of PDFs using free/open-source embeddings and vector DB, with traceable, page-level citations.

- **Repository:** https://github.com/melbinjp/citescope
- **Stack:** FastAPI · `BAAI/bge-m3` · Qdrant · Google Gemini (configurable) · React + Vite
- **Runs on:** CPU only, zero GPU, open-source components (LLM provider is the only hosted/configurable piece)

---

## 1. One-line summary

CiteScope is a Retrieval-Augmented Generation chatbot that answers natural-language questions over a pre-ingested corpus of large PDFs and grounds every answer with **source filename + page number**, while visualizing the ingestion pipeline and the exact passages that were retrieved.

---

## 2. Challenge requirement compliance

| # | Challenge requirement | How CiteScope meets it |
|---|---|---|
| **Corpus** | ≥10 PDFs, each ≥200 pages | Repeatable corpus ingestion procedure (`backend/ingestion/run.py`) walks a directory of 10–100 PDFs (200–2000 pages each) and persists them to Qdrant. |
| **Native extraction** | Extract native text | PyMuPDF per-page extraction with page-number association (`ingestion/extract.py`). |
| **OCR** | OCR scanned pages & embedded images | Targeted Tesseract OCR: full-page OCR only when a page yields zero native characters; embedded-image OCR wrapped via Semantic Markdown Wrapping (`> [System Note: Image Content] …`). |
| **Clean / normalize** | Clean, remove headers/footers, detect language | Whitespace normalization, control-char stripping, repeated header/footer removal (lines on ≥50% of pages), ISO-639-1 language detection with confidence threshold (`ingestion/clean.py`, `ingestion/language.py`). |
| **Chunking** | 500–1000 tokens, 10–30% overlap | Deterministic recursive chunker: ≤800-token passages, ~150-token overlap (kept within 10–30% of chunk size), paragraph→sentence boundaries (`ingestion/chunker.py`). |
| **Metadata** | PDF id, filename, page, (bbox) | Every chunk carries `pdf_id`, `filename`, `page_number` (page of first token for cross-page chunks), `chunk_position`, `language`. |
| **Embedding** | Free/open-source, persisted | `BAAI/bge-m3` (open-source) produces **dense + sparse** vectors in a single forward pass; persisted to Qdrant. |
| **Indexing & retrieval** | ANN index, top-K + scores | Qdrant **HNSW** ANN index over dense vectors; hybrid dense+sparse retrieval; returns top-K candidates with text, fused score, and metadata. |
| **Reranking** | Optional lightweight reranker | Optional token-overlap reranker with stable tie-breaking and timeout fallback to fused order (`reranker.py`). |
| **Generation** | LLM with provenance + citations | Single-pass Chain-of-Thought generation grounded only in retrieved chunks; inline `[filename, page X]` plus one structured citation per source PDF (`generation.py`). |
| **Latency** | 2–5 s | Single hybrid query (server-side RRF) + single-pass generation; SSE streaming for fast first token. Precomputed embeddings remove per-query embedding cost on the corpus. |
| **Evaluation** | p95 latency, R@k, MRR, hallucination, citation accuracy | `EvaluationService` records per-query latency and computes p95, Recall@k, MRR, citation accuracy, and hallucination rate (`evaluation.py`, exposed at `GET /metrics`). |
| **Explainability** | Sources per answer | Page-level citations + retrieved-passage inspector in the UI. |
| **Reproducibility** | Precomputed embeddings, deterministic chunking | Deterministic chunk boundaries + idempotent upserts (SHA-256 point id) → re-ingesting the same corpus produces no duplicates. |
| **Demo deliverable** | Ingestion viz, retrieval viz, answer + citations | Single-page UI shows the pipeline stages (PDF → chunking → embedding → Vector Store), the retrieved chunks with scores, and the final answer with citations. |

---

## 3. Architecture

```
Browser
  └─► Frontend (React + Vite, single-page chat)
        │   • streamed answer  • citation list
        │   • retrieved-passage inspector  • pipeline visualization
        └─► Backend (FastAPI)
              ├─► Embedding: BAAI/bge-m3   (dense + sparse, one pass, CPU)
              ├─► Vector store: Qdrant      (HNSW ANN + sparse, server-side RRF)
              └─► LLM provider: Gemini      (configurable / hosted)
```

Two execution paths:

**A. Offline ingestion (not latency-bound)**
```
PDF → PyMuPDF extract → (targeted Tesseract OCR) → clean / normalize / header-footer removal
    → language detection → deterministic recursive chunking (+ metadata)
    → bge-m3 embed (dense + sparse) → idempotent upsert into Qdrant
    → structured JSON-Lines error log
```

**B. Interactive query (≤5 s; ≤2 s first token streaming)**
```
query → validate → embed (dense + sparse) → Qdrant hybrid search (RRF fusion)
      → optional rerank → single-pass Chain-of-Thought generation
      → stream answer + citations + retrieved chunks to UI
```

---

## 4. Why the stack satisfies "free / open-source"

| Component | Choice | Open-source? |
|---|---|---|
| Embeddings | `BAAI/bge-m3` | Yes (MIT) — dense **and** sparse from one model |
| Vector DB | Qdrant | Yes (Apache-2.0), self-hosted in Docker |
| Extraction | PyMuPDF | Yes |
| OCR | Tesseract | Yes |
| Tokenizer | tiktoken | Yes |
| API | FastAPI | Yes |
| Frontend | React + Vite | Yes |
| LLM | Gemini (default) | Hosted, **configurable** — the challenge explicitly allows open-source or hosted LLMs; the provider is abstracted so any provider can be swapped in |

Everything except the answer-generation LLM runs locally on CPU with no GPU.

---

## 5. Key design decisions

- **Hybrid retrieval with Reciprocal Rank Fusion (RRF):** combines dense semantic ANN and sparse lexical matching. RRF is scale-free (no need to normalize incomparable score scales) and is executed **server-side in a single Qdrant Query API call**, keeping latency low.
- **`bge-m3` for dense + sparse in one pass:** avoids running a separate BM25 stage; supports 8192-token inputs and 100+ languages on CPU.
- **Targeted OCR:** OCR is expensive, so it runs only on zero-text pages and embedded-image bounding boxes — keeping ingestion tractable while still recovering scanned/visual content.
- **Semantic Markdown Wrapping** instead of a vision-language model: OCR'd image text is wrapped in a system-note marker and appended to the page, so a text-only LLM can reference diagrams — preserving the zero-GPU constraint.
- **Deterministic chunking + idempotent upserts:** reproducible chunks and citations; safe re-ingestion with no duplicates.
- **Single-pass Chain-of-Thought (no agentic/multi-step routing):** meets the latency budget while still reasoning over the retrieved context.

---

## 6. Evaluation & monitoring

The `EvaluationService` (exposed at `GET /metrics`) records and computes:

- **p95 latency** — 95th percentile over recorded end-to-end response times (reported once ≥30 samples exist; otherwise an "insufficient sample" result).
- **Recall@k** and **MRR** — over labeled evaluation query sets (≥10 queries).
- **Citation accuracy** — fraction of generated citations whose page is among the reference answer's labeled source pages.
- **Hallucination rate** — fraction of answers containing a claim not matchable to any retrieved chunk.

Each reported value is annotated with the sample size it was computed over.

---

## 7. API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/conversations` | Start a conversation session |
| `POST` | `/conversations/{id}/query` | Ask a question (JSON or SSE stream) |
| `DELETE` | `/conversations/{id}` | Clear conversation history |
| `GET` | `/metrics` | Latest evaluation metrics |
| `GET` | `/healthz` | Liveness + Qdrant reachability + corpus count |

Query body: `{ "q": "your question", "top_k": 5, "stream": true }`
SSE events: `token` · `sources` · `chunks` · `end`

---

## 8. Running the demo

### Prerequisites
- Docker Desktop, Node.js 20+, a Gemini API key

### Steps
```bash
# 1. Configure: create .env at repo root
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-gemini-key
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=corpus
CORPUS_DIR=/data/corpus
SKIP_INGESTION=false
VITE_API_URL=http://localhost:7860
CORPUS_HOST_DIR=./corpus

# 2. Add PDFs to ./corpus/  (10+ PDFs, 200+ pages each for the full challenge)

# 3a. One command:
docker compose up --build
#   Frontend  → http://localhost:80
#   Backend   → http://localhost:7860
#   Qdrant UI → http://localhost:6333/dashboard

# 3b. Or backend+Qdrant in Docker, frontend via Vite (see README)
```

The backend ingests the corpus on startup (idempotent), then the UI is ready for questions.

---

## 9. Suggested demo flow (for judges)

1. **Show the corpus ingestion** — point at `./corpus/` with the PDFs; show `GET /healthz` returning the corpus chunk count (proves the persistent pre-ingested corpus).
2. **Show the pipeline visualization** in the UI: PDF → chunking → embedding → Vector Store.
3. **Ask a question** — watch the answer stream in.
4. **Show the citations** — each answer lists source filename(s) + page numbers; inline `[filename, page X]` tags appear in the text.
5. **Show the retrieved passages** — the inspector lists the top-K chunks with relevance scores, filenames, and pages (retrieval transparency).
6. **Ask a follow-up** — demonstrates multi-turn conversation resolving references to earlier turns.
7. **Show `GET /metrics`** — latency and quality metrics with sample sizes.

---

## 10. Testing

```bash
cd backend
python -m pytest tests/ -v
```
Includes unit tests, property-based tests (Hypothesis), and integration tests across the ingestion and query paths.

---

## 11. Limitations & future work

- **Latency** depends on hardware; the 2–5 s target assumes the relevant context is within the top-K chunks and the LLM provider responds promptly. Streaming delivers a fast first token regardless.
- **First startup** downloads the `bge-m3` model (~2 GB); subsequent starts reuse the cached model.
- **Cross-encoder reranking** ships as a lightweight token-overlap implementation; a full cross-encoder can be swapped in where compute allows.
- **Future:** bounding-box-level highlighting in the source PDF, larger labeled eval sets for tighter metric reporting, and additional open-source LLM providers behind the existing provider abstraction.

---

## 12. Repository

**https://github.com/melbinjp/citescope** — full source for backend, frontend, ingestion pipeline, tests, and Docker deployment.
