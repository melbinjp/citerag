# Implementation Plan: RAG PDF Chatbot

## Overview

This plan converts the design into incremental Python coding steps. The backend is built bottom-up: shared data models first, then pure-logic ingestion components (cleaning, language detection, chunking) that are heavily property-tested, then I/O-bound components (extraction/OCR, embedding, vector store), then the ingestion orchestrator. The query path (retrieval, reranking, generation), cross-cutting services (conversation, evaluation), and the FastAPI surface are wired next, followed by the single-page Web UI, Docker packaging, and end-to-end/performance validation.

Property-based tests use **Hypothesis** (Python) and **fast-check** (frontend), run a minimum of 100 iterations, and are each tagged with `# Feature: rag-pdf-chatbot, Property {n}: {text}`. Each property test lives in its own test module so independent property tests can run in parallel. Sub-tasks marked with `*` (tests) are optional and can be skipped for a faster MVP.

## Tasks

- [x] 1. Set up backend package structure, configuration, and core data models
  - Create `backend/` package layout (`ingestion/`, `providers/`, `tests/`) and add new dependencies to `requirements.txt` (qdrant-client, FlagEmbedding/bge-m3, PyMuPDF, pytesseract, langdetect, hypothesis)
  - Implement `backend/data_models.py` with frozen dataclasses: `ChunkMetadata`, `Chunk`, `Candidate`, `Citation`, `Turn`, `GenerationResult`, `IngestionErrorEntry`, `SparseVector`, `EmbeddingResult`
  - Implement `backend/config.py` loading env settings with range validation: `LLM_PROVIDER`, `EMBEDDING_MODEL`, `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS` (10%–30% of max), `CHUNK_MIN_FINAL_TOKENS` (1..max), `DEFAULT_TOP_K` (1..100), `RERANK_ENABLED`, `RERANK_TIMEOUT_S`, `GENERATION_TIMEOUT_S` (1..120), `QDRANT_URL`, `QDRANT_COLLECTION`, `CORPUS_DIR`, `INGESTION_ERROR_LOG`
  - _Requirements: 4.1, 8.6, 14.3, 14.4_

  - [ ]* 1.2 Write unit tests for configuration validation
    - Test in-range and out-of-range values for chunk/top_k/timeout settings
    - _Requirements: 3.10, 6.3, 8.8_

- [x] 2. Implement text cleaning and language detection
  - [x] 2.1 Implement Cleaner (`backend/ingestion/clean.py`)
    - Whitespace normalization (collapse runs to single space, trim), control-character stripping (U+0000–U+001F and U+007F except U+000A), repeated header/footer line removal (lines appearing on ≥50% of pages in PDFs with ≥3 pages)
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ]* 2.2 Write property test for whitespace normalization
    - **Property 1: Whitespace normalization is collapsed, trimmed, and idempotent**
    - **Validates: Requirements 2.1**

  - [ ]* 2.3 Write property test for control-character stripping
    - **Property 2: Control characters are stripped except newline**
    - **Validates: Requirements 2.2**

  - [ ]* 2.4 Write property test for repeated header/footer removal
    - **Property 3: Repeated header/footer lines are removed**
    - **Validates: Requirements 2.3**

  - [x] 2.5 Implement Language_Detector (`backend/ingestion/language.py`)
    - Assign ISO 639-1 two-letter lowercase code when text ≥20 chars and confidence ≥0.50; assign `und` otherwise
    - _Requirements: 2.4, 2.5_

  - [ ]* 2.6 Write property test for language assignment
    - **Property 4: Language assignment respects threshold and format**
    - **Validates: Requirements 2.4, 2.5**

- [x] 3. Implement deterministic recursive chunking with metadata
  - [x] 3.1 Implement ChunkerConfig, tokenizer, and config validation (`backend/ingestion/chunker.py`)
    - Single fixed tokenizer; `ChunkerConfig` validated on construction, raising `ChunkerConfigError` for out-of-range max/overlap/min_final values and producing no chunks
    - _Requirements: 3.10_

  - [x] 3.2 Implement Chunker.chunk recursive splitting with overlap, metadata, and final-remainder disposition
    - Recursive split at paragraph then sentence boundaries into ≤800-token passages with ~150-token overlap; attach `ChunkMetadata`; record page of first token for spanning chunks; `_emit_final` handles remainder per min-final policy; zero-token input yields zero chunks with a no-content indication
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.9_

  - [ ]* 3.3 Write property test for maximum chunk size
    - **Property 5: Chunks never exceed the configured token size**
    - **Validates: Requirements 3.1, 3.7**

  - [ ]* 3.4 Write property test for consecutive-chunk overlap bounds
    - **Property 6: Consecutive-chunk overlap stays within bounds**
    - **Validates: Requirements 3.2**

  - [ ]* 3.5 Write property test for chunk metadata completeness
    - **Property 7: Every chunk carries complete and correct metadata**
    - **Validates: Requirements 3.3, 3.4**

  - [ ]* 3.6 Write property test for chunking determinism
    - **Property 8: Chunking is deterministic**
    - **Validates: Requirements 3.5, 5.4**

  - [ ]* 3.7 Write property test for full-content coverage
    - **Property 9: Chunking covers all content with no tokens dropped**
    - **Validates: Requirements 3.6, 3.7, 3.8**

  - [ ]* 3.8 Write property test for invalid-config rejection
    - **Property 10: Invalid chunker configuration is rejected without side effects**
    - **Validates: Requirements 3.10**

  - [ ]* 3.9 Write property test for per-page chunk coverage
    - **Property 16: Every extracted non-empty page is represented by at least one chunk**
    - **Validates: Requirements 5.3**

  - [ ]* 3.10 Write property test for final-remainder policy
    - **Property 39: The final remainder is placed per the minimum-final-chunk policy**
    - **Validates: Requirements 3.6, 3.7, 3.8**

- [x] 4. Checkpoint - Ensure ingestion pure-logic tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement PDF text extraction and targeted OCR
  - [x] 5.1 Implement Text_Extractor and OCR routing (`backend/ingestion/extract.py`)
    - PyMuPDF native per-page extraction with page-number association; route a page to full-page OCR only when native extraction yields zero characters; OCR embedded image bounding boxes and append via Semantic Markdown Wrapping (`> [System Note: Image Content] ...`); support 1–2000 page documents
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 5.2 Write property test for image-text wrapping
    - **Property 13: Image-derived text is wrapped and appended identifiably**
    - **Validates: Requirements 1.3, 1.4**

  - [ ]* 5.3 Write property test for OCR routing
    - **Property 14: A page is routed to OCR exactly when it yields zero native characters**
    - **Validates: Requirements 1.2**

  - [ ]* 5.4 Write integration tests for PyMuPDF and Tesseract on fixture PDFs
    - Native extraction + page association on a digital fixture; OCR on a scanned page and on an embedded image
    - _Requirements: 1.1, 1.2, 1.3, 1.5_

- [x] 6. Implement the structured ingestion error log
  - [x] 6.1 Implement Ingestion_Error_Log writer/reader (`backend/ingestion/error_log.py`)
    - Serialize/parse `IngestionErrorEntry` as JSON Lines (one self-contained record per line) with filename, optional page number, stage (extraction/ocr/embedding/persistence), reason, ISO-8601 timestamp; flush to durable storage on run completion
    - _Requirements: 13.1, 13.2, 13.3, 13.5_

  - [ ]* 6.2 Write property test for error-log round-trip
    - **Property 36: Error-log entries round-trip through the JSON Lines format**
    - **Validates: Requirements 13.3**

- [x] 7. Implement the embedding model wrapper
  - [x] 7.1 Implement EmbeddingModel wrapper (`backend/embedding.py`)
    - Load `BAAI/bge-m3` once; return exactly one `EmbeddingResult` (dense + sparse) per input text in input order from a single forward pass
    - _Requirements: 4.1, 4.2_

  - [ ]* 7.2 Write property test for embedding output shape and order
    - **Property 11: Embedding produces exactly one dense and one sparse vector per chunk, order preserved**
    - **Validates: Requirements 4.2**

  - [ ]* 7.3 Write integration test for bge-m3 dense+sparse output
    - Assert dense dimension 1024 and a populated sparse vector in one pass
    - _Requirements: 4.1_

- [x] 8. Implement the Qdrant vector store adapter
  - [x] 8.1 Implement VectorStore (`backend/vector_store.py`)
    - `ensure_collection` (named `dense` HNSW + `sparse` vectors), deterministic point id = `sha256(normalized_text + metadata)`, idempotent `upsert_chunk`, `exists`, `hybrid_search` (single Query API call: dense prefetch + sparse prefetch fused via RRF), `count`; durable persistence config
    - _Requirements: 4.3, 4.4, 4.5, 6.1_

  - [ ]* 8.2 Write property test for point-id determinism and idempotency
    - **Property 15: Point ids are deterministic and idempotent**
    - **Validates: Requirements 4.4, 5.4**

  - [ ]* 8.3 Write integration test for persistence across restart
    - Upsert → simulate restart → retrieve, confirming vectors/text/metadata survive
    - _Requirements: 4.3, 4.5, 14.5_

- [x] 9. Implement the ingestion pipeline orchestrator
  - [x] 9.1 Implement Ingestion_Pipeline (`backend/ingestion/pipeline.py`)
    - Orchestrate extraction → cleaning → language → chunking → embedding (reusing existing vectors for identical chunk text+metadata) → idempotent upsert; continue-on-failure at every stage logging exactly one structured error per failure; ensure ≥1 chunk per successfully extracted non-empty page
    - _Requirements: 1.6, 1.7, 4.4, 4.6, 4.7, 5.3, 5.5, 13.4_

  - [ ]* 9.2 Write property test for fault isolation during ingestion
    - **Property 12: A failure at any stage is logged once and never aborts the batch**
    - **Validates: Requirements 1.6, 1.7, 4.6, 4.7, 5.5, 13.1, 13.2, 13.4**

  - [ ]* 9.3 Write property test for ingestion summary partitioning
    - **Property 17: Ingestion summary partitions the input**
    - **Validates: Requirements 5.6**

  - [x] 9.4 Implement repeatable corpus ingestion procedure (`backend/ingestion/run.py`)
    - Walk `CORPUS_DIR` (10–100 PDFs, 200–2000 pages each), invoke the pipeline, persist the error log, and report ingested count + failed filenames; idempotent re-runs produce no duplicates
    - _Requirements: 5.1, 5.2, 5.4, 5.6_

- [x] 10. Checkpoint - Ensure ingestion path tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement the retrieval service
  - [x] 11.1 Implement RetrievalService (`backend/retrieval.py`)
    - Validate query (1–4000 chars, top_k 1–100, default 5) before embedding; embed query (dense+sparse); single hybrid search fused via RRF; assemble candidates with text, fused score, and metadata; empty result when scope has no chunks
    - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.6, 5.7_

  - [ ]* 11.2 Write property test for RRF fusion
    - **Property 18: RRF fusion is bounded, ordered, and faithful**
    - **Validates: Requirements 6.3**

  - [ ]* 11.3 Write property test for candidate completeness
    - **Property 19: Every retrieval candidate is complete**
    - **Validates: Requirements 6.4**

  - [ ]* 11.4 Write property test for invalid-query rejection
    - **Property 20: Invalid queries are rejected before embedding**
    - **Validates: Requirements 6.6**

- [x] 12. Implement optional reranking
  - [x] 12.1 Implement Reranker (`backend/reranker.py`)
    - Stable reorder by descending relevance with original-rank tie-breaking; metadata-preserving permutation; timeout/error fallback to fused order; pass-through when disabled
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 12.2 Write property test for rerank ordering and tie-breaking
    - **Property 21: Reranking sorts by relevance with stable tie-breaking**
    - **Validates: Requirements 7.1, 7.4**

  - [ ]* 12.3 Write property test for rerank permutation invariance
    - **Property 22: Reranking is a metadata-preserving permutation**
    - **Validates: Requirements 7.3**

  - [ ]* 12.4 Write property test for disabled/failed rerank fallback
    - **Property 23: Disabled or failed reranking preserves hybrid order**
    - **Validates: Requirements 7.2, 7.5**

- [x] 13. Implement LLM providers and the answer generator
  - [x] 13.1 Implement LLMProvider abstraction and GeminiProvider (`backend/providers/`)
    - `LLMProvider` protocol with async `generate(prompt, timeout_s, stream)`; default `GeminiProvider`; provider selection from config
    - _Requirements: 8.6, 14.4_

  - [x] 13.2 Implement AnswerGenerator (`backend/generation.py`)
    - Single-pass CoT prompt with Top_K chunks (no agentic routing, no multi-step); inline `[filename, page X]` provenance; one structured Citation per distinct filename with sorted unique pages; answer-language matching; empty-chunk → "no relevant information" + zero citations; provider error/timeout → error result with no partial answer + zero citations; reference image markers
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.7, 8.8, 8.9_

  - [ ]* 13.3 Write property test for single provider invocation
    - **Property 24: The Answer_Generator invokes the configured provider exactly once**
    - **Validates: Requirements 8.2, 8.6**

  - [ ]* 13.4 Write property test for citation aggregation
    - **Property 25: Structured citations aggregate one entry per source PDF**
    - **Validates: Requirements 8.4**

  - [ ]* 13.5 Write property test for safe empty/error responses
    - **Property 26: Empty context and generation errors yield safe, citation-free responses**
    - **Validates: Requirements 8.5, 8.8, 8.9**

  - [ ]* 13.6 Write integration test for single-pass CoT with a real provider
    - Assert exactly one provider call, prompt contains injected chunks, and answer language matches a non-English query
    - _Requirements: 8.1, 8.3, 8.7_

- [x] 14. Checkpoint - Ensure query-path tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Implement the conversation session store
  - [x] 15.1 Implement ConversationStore (`backend/conversation.py`)
    - Per-session ordered turns; append with oldest-turn eviction to satisfy 10-turn and 4000-token bounds; fast-path skip of cleanup when session was empty; `clear`; per-session isolation
    - _Requirements: 12.1, 12.4, 12.5, 12.6, 12.7_

  - [ ]* 15.2 Write property test for bounded recent-turn retention
    - **Property 32: Conversation history is bounded and retains the most recent turns in order**
    - **Validates: Requirements 12.1, 12.4**

  - [ ]* 15.3 Write property test for clearing a conversation
    - **Property 33: Clearing a conversation removes all prior context**
    - **Validates: Requirements 12.6**

  - [ ]* 15.4 Write property test for per-session isolation
    - **Property 34: Conversation histories are isolated per session**
    - **Validates: Requirements 12.7**

  - [ ]* 15.5 Write property test for history incorporation into formulation
    - **Property 35: Non-empty history is incorporated into query formulation**
    - **Validates: Requirements 12.2, 12.8**

  - [ ]* 15.6 Write property test for first-turn cleanup skip
    - **Property 40: Adding the first turn to an empty session skips history cleanup**
    - **Validates: Requirements 12.5**

- [x] 16. Implement the evaluation service
  - [x] 16.1 Implement EvaluationService (`backend/evaluation.py`)
    - Latency ring buffer recording ms per handled query; p95 (≥30 samples, seconds with ≥2 decimals, else insufficient-sample result); R@k and MRR over labeled sets (≥10 queries); citation-accuracy and hallucination-rate ratios in [0,1]; missing-label error; `report` with latest values annotated by sample size
    - _Requirements: 9.3, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

  - [ ]* 16.2 Write property test for one latency sample per query
    - **Property 27: Exactly one latency sample is recorded per handled query**
    - **Validates: Requirements 9.3**

  - [ ]* 16.3 Write property test for p95 sufficiency gating
    - **Property 28: p95 latency is computed only with a sufficient sample**
    - **Validates: Requirements 10.1, 10.2**

  - [ ]* 16.4 Write property test for R@k and MRR
    - **Property 29: Retrieval quality metrics match their definitions and stay in range**
    - **Validates: Requirements 10.3**

  - [ ]* 16.5 Write property test for citation-accuracy and hallucination-rate
    - **Property 30: Citation-accuracy and hallucination-rate are correct ratios in range**
    - **Validates: Requirements 10.4, 10.5**

  - [ ]* 16.6 Write property test for under-labeled evaluation sets
    - **Property 31: Under-labeled evaluation sets produce an error, not a metric**
    - **Validates: Requirements 10.6**

- [x] 17. Wire the FastAPI HTTP surface
  - [x] 17.1 Implement API endpoints and service wiring (`backend/app.py`)
    - FastAPI lifespan loads embedding model + Qdrant client once; `POST /conversations`, `POST /conversations/{id}/query` (JSON + SSE streaming with `sources`/`chunks` and `end`/`error` events), `DELETE /conversations/{id}`, `GET /metrics`, `GET /healthz` (Qdrant reachability + corpus count); record latency for every handled query including errors
    - _Requirements: 9.3, 12.1, 12.2, 12.3, 12.6, 14.1, 14.5_

  - [ ]* 17.2 Write integration tests for API endpoints
    - Conversation lifecycle, query (JSON + SSE), metrics, healthz, and per-query latency recording on success and error paths
    - _Requirements: 9.3, 12.6, 14.5_

- [x] 18. Checkpoint - Ensure backend services and API tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 19. Implement the single-page Web UI
  - [x] 19.1 Implement the single-page chat App (`frontend/src/App.jsx`)
    - One page with query input + answer area and no full-page reload; pending indicator shown during generation and removed on answer/error; error state displays a could-not-be-answered message while retaining the submitted query text
    - _Requirements: 11.1, 11.7, 11.8_

  - [x] 19.2 Implement CitationList component (`frontend/src/components/CitationList.jsx`)
    - Render each citation's filename and all page numbers; show a no-sources indication when empty
    - _Requirements: 11.2, 11.3_

  - [x] 19.3 Implement RetrievedChunks component (`frontend/src/components/RetrievedChunks.jsx`)
    - Render each Top_K chunk with relevance score, filename, and page; show a no-passages indication when empty
    - _Requirements: 11.4, 11.5_

  - [x] 19.4 Implement PipelineStages component (`frontend/src/components/PipelineStages.jsx`)
    - Static ordered display: PDF → chunking → embedding → Vector_Store
    - _Requirements: 11.6_

  - [x] 19.5 Extend the API client SSE handling (`frontend/src/services/api.js`)
    - Consume streamed answer tokens plus `sources`/`chunks` and `end`/`error` events
    - _Requirements: 11.2, 11.4, 11.7, 11.8_

  - [ ]* 19.6 Write property test for citation rendering completeness
    - **Property 37: Rendered citations show every filename and page**
    - **Validates: Requirements 11.2**

  - [ ]* 19.7 Write property test for retrieved-chunk rendering completeness
    - **Property 38: Rendered retrieved chunks show score, filename, and page for every chunk**
    - **Validates: Requirements 11.4**

  - [ ]* 19.8 Write UI interaction tests
    - No-sources and no-passages indications, pipeline-stage order, pending indicator appears then is removed on answer/error, metrics report annotated with sample sizes
    - _Requirements: 11.3, 11.5, 11.6, 11.7, 11.8, 10.7_

- [x] 20. Package the Docker deployment
  - [x] 20.1 Create backend and frontend Docker deployment
    - Backend `Dockerfile` + `docker-compose.yml` running RAG API and Qdrant with a persistent volume that reloads the corpus on start; frontend `Dockerfile`; CPU-only open-source stack with configurable hosted LLM provider
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [ ]* 20.2 Write smoke tests for deployment
    - `docker compose up` starts API + Qdrant + frontend, HNSW collection created, embedding model loads on CPU without CUDA, provider env toggle selects configured provider
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [x] 21. End-to-end integration and performance validation
  - [x] 21.1 Wire end-to-end flow over a fixture corpus
    - Ingest a small fixture corpus, then answer queries without upload through the full retrieve → (rerank) → generate → cite path
    - _Requirements: 5.7, 6.2, 8.1, 9.1_

  - [ ]* 21.2 Write end-to-end integration test
    - Ingest fixture corpus then query without upload, asserting answer + citations + retrieved chunks
    - _Requirements: 5.1, 5.2, 5.7_

  - [ ]* 21.3 Write performance tests for latency SLAs
    - Hybrid search ≤1 s; non-streaming end-to-end ≤5 s and p95 ≤5 s over ≥30 queries; streaming first token ≤2 s
    - _Requirements: 6.7, 9.1, 9.2, 9.4_

- [x] 22. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (unit, property, integration, smoke, performance tests) and can be skipped for a faster MVP.
- Each task references specific requirement sub-clauses for traceability.
- Property-based tests use Hypothesis (backend) and fast-check (frontend), run ≥100 iterations, and are tagged with the design property number; each property is implemented as a single property test in its own module.
- Checkpoints ensure incremental validation; integration/smoke/performance tests cover external dependencies (PyMuPDF, Tesseract, Qdrant, LLM provider) and SLAs that are not expressible as universal properties.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "2.5", "3.1", "5.1", "6.1", "7.1", "8.1", "12.1", "13.1", "15.1", "16.1", "19.2", "19.3", "19.4", "19.5"] },
    { "id": 2, "tasks": ["3.2", "11.1", "13.2", "19.1", "2.2", "2.3", "2.4", "2.6", "5.2", "5.3", "5.4", "6.2", "7.2", "7.3", "8.2", "8.3", "12.2", "12.3", "12.4", "15.2", "15.3", "15.4", "15.5", "15.6", "16.2", "16.3", "16.4", "16.5", "16.6", "19.6", "19.7"] },
    { "id": 3, "tasks": ["9.1", "17.1", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "11.2", "11.3", "11.4", "13.3", "13.4", "13.5", "13.6", "19.8"] },
    { "id": 4, "tasks": ["9.2", "9.3", "9.4", "17.2", "20.1"] },
    { "id": 5, "tasks": ["21.1", "20.2"] },
    { "id": 6, "tasks": ["21.2", "21.3"] }
  ]
}
```
