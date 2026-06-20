# Requirements Document

## Introduction

This feature delivers an improved Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions from a large private corpus of PDF documents. It builds on and improves the existing DocQA backend (FastAPI + Gemini) and adds a new single-page web frontend that replaces the current multi-page React application.

The baseline target is the hackathon Challenge 1 minimum: a corpus of at least 10 PDFs, each at least 200 pages, ingested into a persistent open-source vector database using open-source embeddings, with end-to-end query latency in the 2–5 second range. Answers must include provenance (source PDF filename and page number). The system improves on the current ephemeral, per-session design by adding a pre-ingested persistent corpus, targeted OCR for scanned pages and embedded images, page-level chunk metadata, an approximate nearest neighbor (ANN) index, deterministic recursive chunking, hybrid dense-plus-sparse retrieval, multi-turn conversation, and evaluation/monitoring of retrieval and answer quality. The new single-page UI provides a chat interface, citation display, and retrieval visualization.

The finalized open-source stack is: PyMuPDF for fast native text extraction with conditional, targeted Tesseract OCR; `BAAI/bge-m3` as the embedding model, which emits both a dense semantic vector and a sparse lexical vector and supports up to 8192 tokens across 100+ languages; Qdrant as the persistent vector store running in Docker; hybrid retrieval that fuses dense ANN (HNSW) results and sparse lexical results using Reciprocal Rank Fusion (RRF); and single-pass Chain-of-Thought answer generation. Text recovered by OCR from embedded images and diagrams is integrated using Semantic Markdown Wrapping rather than a vision-language model, keeping the system within a zero-VRAM, CPU-only, open-source constraint. The entire system — backend (RAG API plus Qdrant) and frontend — is deployable via Docker and runs without any GPU. The answer-generation LLM provider remains configurable; the existing hosted Gemini provider is allowed, as the hackathon permits open-source or hosted LLMs.

Document ingestion is an offline, precomputed step. It is not bound by the 2–5 second end-to-end query latency budget, which applies only to interactive query answering.

## Glossary

- **System**: The complete RAG PDF chatbot solution, consisting of the Ingestion_Pipeline, the Retrieval_Service, the Answer_Generator, the Evaluation_Service, and the Web_UI.
- **Ingestion_Pipeline**: The backend component that extracts, OCRs, cleans, chunks, embeds, and persists PDF content into the Vector_Store. The Ingestion_Pipeline runs offline and is not bound by the interactive query latency budget.
- **Text_Extractor**: The PyMuPDF-based Ingestion_Pipeline subcomponent that extracts native (digital) text from PDF pages.
- **OCR_Engine**: The open-source optical character recognition subcomponent (for example Tesseract) that extracts text in a targeted manner from embedded image bounding boxes and from full pages that yield zero native characters.
- **Semantic_Markdown_Wrapping**: The technique by which text recovered by the OCR_Engine from an embedded image or diagram is wrapped with a system note marker (for example `> [System Note: Image Content] ...`) and appended to the originating page's text, so that the Answer_Generator can treat that text as visual-origin content without a vision-language model.
- **Language_Detector**: The subcomponent that identifies the primary language of extracted text.
- **Chunker**: The recursive text-splitting subcomponent that splits normalized text at logical boundaries into token-bounded passages with overlap and attaches metadata.
- **Embedding_Model**: The open-source model `BAAI/bge-m3`, which supports inputs up to 8192 tokens across 100+ languages and produces, in a single forward pass, both a Dense_Vector and a Sparse_Vector for a given text.
- **Dense_Vector**: The dense semantic embedding produced by the Embedding_Model, used for approximate nearest neighbor similarity search.
- **Sparse_Vector**: The sparse lexical (BM25-style) embedding produced by the Embedding_Model, used for lexical matching.
- **Vector_Store**: The persistent open-source vector database Qdrant, which runs in Docker, stores Dense_Vectors, Sparse_Vectors, raw chunk text, and Chunk_Metadata, provides the ANN_Index, and retains data across restarts.
- **ANN_Index**: The HNSW approximate nearest neighbor index maintained by the Vector_Store over the persisted Dense_Vectors for fast similarity search.
- **Retrieval_Service**: The backend component that embeds a query into a Dense_Vector and a Sparse_Vector, performs a Hybrid_Search in the Vector_Store, and returns ranked candidate chunks with metadata and scores.
- **Hybrid_Search**: A single Vector_Store query that combines dense ANN (HNSW) results and sparse lexical (BM25-style) results and fuses them into one ranked list using Reciprocal Rank Fusion.
- **RRF (Reciprocal Rank Fusion)**: The rank-based fusion method that combines the dense and sparse result rankings of a Hybrid_Search into a single ranked candidate list.
- **Reranker**: The optional subcomponent that re-orders retrieved candidate chunks using a cross-encoder or lexical filter.
- **Answer_Generator**: The component that synthesizes an answer from a query and retrieved chunks in a single-pass Chain-of-Thought generation using a configurable LLM provider.
- **Chain-of-Thought (CoT)**: The single-pass prompting strategy in which the Top_K retrieved chunks are injected into one system prompt that instructs the LLM_Provider to reason step by step before producing the final answer, without agentic query routing or multi-step generation.
- **LLM_Provider**: The configurable large language model service (hosted such as Gemini, or open-source) used by the Answer_Generator.
- **Conversation_Session**: An ordered history of turns within a single user session, where each turn comprises a user question and the corresponding assistant answer. Conversation_Session history is scoped to that session and is not shared across sessions.
- **Evaluation_Service**: The component that records and reports latency and retrieval/answer quality metrics.
- **Web_UI**: The new single-page web frontend providing the chat interface, citation display, and retrieval visualization.
- **Chunk_Metadata**: The per-chunk data set comprising PDF identifier, filename, page number, and chunk position.
- **Citation**: A provenance reference attached to an answer, consisting of a source PDF filename and one or more page numbers.
- **Ingestion_Error_Log**: The structured, machine-readable log (for example JSON Lines) to which the Ingestion_Pipeline writes every extraction, OCR, embedding, and persistence failure, intended for consumption by an automated agent.
- **Corpus**: The pre-ingested set of PDF documents persisted in the Vector_Store.
- **Top_K**: The configurable number of candidate chunks returned by the Retrieval_Service for a query.
- **p95_Latency**: The 95th percentile of measured end-to-end query response times.
- **R@k**: Recall at k, the fraction of relevant chunks present in the top-k retrieved results for an evaluation query.
- **MRR**: Mean Reciprocal Rank of the first relevant chunk across evaluation queries.

## Requirements

### Requirement 1: PDF Text and Targeted OCR Extraction

**User Story:** As a knowledge worker, I want the system to extract text from both digital and scanned PDFs, so that content from every page is searchable regardless of how the PDF was produced.

#### Acceptance Criteria

1. WHEN a PDF page contains a native digital text layer, THE Text_Extractor SHALL extract all native text content of that page using PyMuPDF and SHALL associate the extracted text with that page's page number.
2. IF a PDF page yields zero native characters, THEN THE OCR_Engine SHALL extract text from a rendered image of the full page using an open-source OCR engine and SHALL associate the extracted text with that page's page number.
3. WHERE a PDF page contains embedded raster images alongside native text, THE OCR_Engine SHALL extract text from the bounding box of each embedded image, and THE Ingestion_Pipeline SHALL append that image-derived text to the same page's text using Semantic_Markdown_Wrapping.
4. WHEN the OCR_Engine extracts text from an embedded image, THE Ingestion_Pipeline SHALL wrap that text with the Semantic_Markdown_Wrapping system note marker so that the text is identifiable as visual-origin content.
5. THE Ingestion_Pipeline SHALL process PDF documents containing 1 to 2000 pages each.
6. IF text extraction and OCR both yield zero characters for a PDF document, THEN THE Ingestion_Pipeline SHALL record an ingestion error identifying the PDF filename and SHALL continue ingesting the remaining PDF documents without interruption.
7. IF a PDF document cannot be opened because it is corrupted, password-protected, or not a valid PDF, THEN THE Ingestion_Pipeline SHALL record an ingestion error identifying the PDF filename and the failure reason, SHALL skip that document, and SHALL continue ingesting the remaining PDF documents without interruption.

### Requirement 2: Text Cleaning, Normalization, and Language Detection

**User Story:** As a knowledge worker, I want extracted text to be cleaned and language-tagged, so that retrieval quality is high and multilingual content is handled correctly.

#### Acceptance Criteria

1. WHEN text is extracted from a PDF page, THE Ingestion_Pipeline SHALL replace every sequence of one or more consecutive whitespace characters in that page's extracted text with a single space character and remove leading and trailing whitespace from that page's text.
2. WHEN text is extracted from a PDF page, THE Ingestion_Pipeline SHALL remove all Unicode control characters in the ranges U+0000 through U+001F and U+007F, except the line feed character (U+000A), from that page's text.
3. WHEN an identical line, compared after whitespace normalization, appears on at least 50 percent of the pages of a PDF that contains 3 or more pages, THE Ingestion_Pipeline SHALL remove every occurrence of that repeated line from each page's text.
4. WHEN a page's normalized text contains at least 20 characters and the Language_Detector determines a primary language with a confidence of at least 0.50, THE Language_Detector SHALL assign to that page a primary language code expressed as an ISO 639-1 two-letter lowercase code.
5. IF a page's normalized text contains fewer than 20 characters, or the Language_Detector cannot determine a primary language with a confidence of at least 0.50, THEN THE Language_Detector SHALL assign the language code `und` to that page's text.

### Requirement 3: Deterministic Recursive Chunking with Metadata

**User Story:** As a developer, I want documents chunked deterministically into token-bounded passages with page metadata, so that retrieval is reproducible and answers can cite exact page numbers.

#### Acceptance Criteria

1. THE Chunker SHALL split normalized text recursively at logical boundaries, attempting paragraph boundaries first and then sentence boundaries, into passages of at most 800 tokens each, where token counts are measured using a single consistent tokenizer.
2. THE Chunker SHALL create an overlap of approximately 150 tokens, rounded to the nearest whole token, between consecutive chunks of the same PDF, where the overlap remains between 10 percent and 30 percent of the configured chunk token size.
3. THE Chunker SHALL attach Chunk_Metadata containing PDF identifier, filename, and page number to every chunk.
4. WHEN a chunk spans content from more than one page, THE Chunker SHALL record the page number of the page contributing the first token of that chunk.
5. WHEN the Chunker processes identical input text and identical configuration values more than once, THE Chunker SHALL produce identical chunk boundaries and identical Chunk_Metadata on each run.
6. WHEN the final remaining text of a PDF contains fewer tokens than the configured chunk token size but at least the configured minimum final chunk token size, THE Chunker SHALL place that remaining text in a single final chunk with its Chunk_Metadata attached.
7. IF the final remaining text of a PDF contains fewer tokens than the configured minimum final chunk token size AND at least one prior chunk exists for that PDF, THEN THE Chunker SHALL merge that remaining text into the immediately preceding chunk of the same PDF, retaining the preceding chunk's Chunk_Metadata.
8. IF the final remaining text of a PDF contains fewer tokens than the configured minimum final chunk token size AND no prior chunk exists for that PDF, THEN THE Chunker SHALL place that remaining text in a single final chunk with its Chunk_Metadata attached.
9. WHEN the normalized input text contains zero tokens, THE Chunker SHALL produce zero chunks and SHALL return an indication that no content was available to chunk.
10. IF the configured chunk token size or overlap value falls outside its permitted range, THEN THE Chunker SHALL reject the configuration with an error and SHALL NOT alter previously produced chunks.

### Requirement 4: Embedding and Persistence to the Vector Store

**User Story:** As a developer, I want chunk embeddings precomputed once and persisted, so that the corpus survives restarts and queries do not pay re-embedding cost.

#### Acceptance Criteria

1. THE Embedding_Model SHALL be the open-source model `BAAI/bge-m3`, and WHEN the Embedding_Model embeds a chunk, THE Embedding_Model SHALL produce both a Dense_Vector and a Sparse_Vector for that chunk in a single forward pass.
2. WHEN a chunk is produced by the Chunker, THE Ingestion_Pipeline SHALL compute exactly one Dense_Vector and exactly one Sparse_Vector for that chunk using the Embedding_Model.
3. WHEN a chunk embedding is computed, THE Vector_Store SHALL persist the Dense_Vector, the Sparse_Vector, the chunk text, and the Chunk_Metadata to durable storage that retains the data across System process restarts.
4. IF a chunk with identical text and identical Chunk_Metadata already exists in the Vector_Store, THEN THE Ingestion_Pipeline SHALL reuse the existing Dense_Vector and Sparse_Vector and SHALL NOT recompute the embeddings for that chunk.
5. WHEN the System restarts after the Corpus has been ingested, THE Vector_Store SHALL make every previously persisted Dense_Vector, Sparse_Vector, chunk text, and Chunk_Metadata available for retrieval without re-ingestion.
6. IF embedding computation for a chunk fails, THEN THE Ingestion_Pipeline SHALL record an embedding error identifying the chunk's filename and page number and SHALL continue processing the remaining chunks without interruption.
7. IF persisting a chunk embedding to the Vector_Store fails, THEN THE Ingestion_Pipeline SHALL record a persistence error identifying the chunk's filename and page number and SHALL leave the previously persisted chunks unchanged.

### Requirement 5: Pre-Ingested Persistent Corpus

**User Story:** As a demo presenter, I want a corpus of at least 10 large PDFs pre-ingested and persisted, so that the chatbot can answer questions immediately without per-session uploads.

#### Acceptance Criteria

1. THE System SHALL provide a repeatable ingestion procedure that loads a Corpus of between 10 and 100 PDF documents into the Vector_Store.
2. THE System SHALL support a Corpus in which each PDF document contains between 200 and 2000 pages.
3. WHEN the ingestion procedure completes, THE Vector_Store SHALL contain at least one persisted chunk for every successfully extracted page of every PDF in the Corpus.
4. WHEN the ingestion procedure is run more than once on an identical source set of PDF documents with identical configuration, THE System SHALL persist identical chunks and Chunk_Metadata with no duplicate chunks in the Vector_Store.
5. IF a PDF document in the source set cannot be loaded or yields no extractable text, THEN THE System SHALL record an ingestion error identifying the PDF filename and SHALL continue ingesting the remaining PDF documents.
6. WHEN the ingestion procedure completes, THE System SHALL report the count of successfully ingested PDF documents and the filenames of any PDF documents that failed.
7. WHILE the Corpus is persisted in the Vector_Store, THE Retrieval_Service SHALL return results drawn from the persisted Corpus chunks without requiring the user to upload documents.

### Requirement 6: Hybrid Retrieval with ANN Indexing and Reciprocal Rank Fusion

**User Story:** As a knowledge worker, I want fast hybrid similarity-and-lexical search over the corpus, so that relevant passages are retrieved within the latency budget even for a large corpus.

#### Acceptance Criteria

1. WHEN the Ingestion_Pipeline persists chunk embeddings to the Vector_Store, THE Vector_Store SHALL build an HNSW ANN_Index over the persisted Dense_Vectors.
2. WHEN the Retrieval_Service receives a query containing between 1 and 4000 characters, THE Retrieval_Service SHALL compute the query Dense_Vector and the query Sparse_Vector using the Embedding_Model.
3. WHEN the query Dense_Vector and query Sparse_Vector are computed, THE Retrieval_Service SHALL perform a single Hybrid_Search in the Vector_Store that combines the dense ANN (HNSW) results and the sparse lexical results and fuses them using RRF, and SHALL return at most Top_K candidate chunks ordered by descending fused score, where Top_K is configurable to an integer between 1 and 100 and defaults to 5 when not specified.
4. WHEN the Retrieval_Service returns a candidate chunk, THE Retrieval_Service SHALL include that chunk's text, fused relevance score, and Chunk_Metadata in the result.
5. IF the Vector_Store contains no chunks for the requested scope, THEN THE Retrieval_Service SHALL return an empty result set.
6. IF the Retrieval_Service receives a query that is empty or exceeds 4000 characters, THEN THE Retrieval_Service SHALL reject the query, SHALL NOT compute a query embedding, and SHALL return an error response indicating that the query is invalid.
7. WHEN the Retrieval_Service performs a Hybrid_Search for a computed query, THE Retrieval_Service SHALL return the ranked candidate chunks within 1 second.

### Requirement 7: Optional Reranking and Filtering

**User Story:** As a knowledge worker, I want retrieved candidates optionally reranked, so that the most relevant passages are prioritized before answer generation.

#### Acceptance Criteria

1. WHERE reranking is enabled, THE Reranker SHALL reorder the Top_K candidate chunks in descending order of a relevance score computed from the query and each candidate chunk.
2. WHERE reranking is disabled, THE Retrieval_Service SHALL pass candidate chunks to the Answer_Generator in the order produced by the Hybrid_Search.
3. WHERE reranking is enabled, THE Reranker SHALL preserve the Chunk_Metadata and chunk text of each candidate chunk after reordering and SHALL return the same number of candidate chunks it received.
4. WHERE reranking is enabled, WHEN two or more candidate chunks have equal relevance scores, THE Reranker SHALL order those chunks by their original Hybrid_Search rank.
5. IF reranking is enabled and the Reranker returns an error or does not produce relevance scores within the configured reranking timeout, THEN THE Retrieval_Service SHALL pass the candidate chunks to the Answer_Generator in the order produced by the Hybrid_Search.

### Requirement 8: Single-Pass Chain-of-Thought Answer Generation with Citations

**User Story:** As a knowledge worker, I want answers grounded in retrieved passages with source citations, so that I can trust and verify each answer.

#### Acceptance Criteria

1. WHEN the Answer_Generator receives a query and a non-empty set of retrieved chunks, THE Answer_Generator SHALL inject the Top_K retrieved chunks into a single Chain-of-Thought system prompt and synthesize an answer in one generation pass using the configured LLM_Provider, derived only from the content of the provided chunks, and SHALL NOT incorporate information sourced from outside the provided chunks.
2. THE Answer_Generator SHALL NOT perform agentic query routing and SHALL NOT perform multi-step or map-reduce generation for a single query.
3. WHEN a retrieved chunk provided to the Answer_Generator contains the Semantic_Markdown_Wrapping image marker, THE Answer_Generator SHALL use that note to contextually reference the corresponding diagram or table in the answer.
4. WHEN the Answer_Generator produces an answer, THE Answer_Generator SHALL append the provenance text `[filename, page X]` immediately after each claim derived from a provided chunk and SHALL also return one structured Citation for each distinct source PDF present in the set of retrieved chunks, where each Citation contains that PDF's filename and the page number of every provided chunk from that PDF.
5. IF the set of retrieved chunks is empty, THEN THE Answer_Generator SHALL return a response stating that no relevant information was found and SHALL attach zero Citations.
6. WHERE the LLM_Provider is configured to a hosted provider, THE Answer_Generator SHALL route the generation request to that configured hosted provider and SHALL NOT use any other provider for that request.
7. WHERE the detected primary language of the query is a language other than English, THE Answer_Generator SHALL produce the answer text in that same detected language.
8. IF the configured LLM_Provider returns an error, OR IF the configured LLM_Provider does not return a complete response within the configured generation timeout (default 30 seconds, configurable in the range 1 to 120 seconds), THEN THE Answer_Generator SHALL return an error response that identifies that answer generation failed.
9. WHEN the Answer_Generator returns an error response for a query, THE Answer_Generator SHALL NOT return a partial or fabricated answer and SHALL attach zero Citations.

### Requirement 9: End-to-End Latency

**User Story:** As a knowledge worker, I want answers returned quickly, so that the chat experience feels responsive.

#### Acceptance Criteria

1. WHILE streaming is disabled, WHEN a user submits a query against the pre-ingested Corpus whose relevant context is fully contained within the configured Top_K retrieved chunks, THE System SHALL return the complete answer with all associated Citations within 5 seconds measured from receipt of the query by the System to delivery of the final answer token to the Web_UI.
2. WHILE streaming is disabled, THE System SHALL maintain a p95_Latency, computed over the recorded end-to-end response times, that does not exceed 5 seconds.
3. THE Evaluation_Service SHALL record, in milliseconds, the end-to-end response time of every query measured from receipt of the query by the System to delivery of the final answer token or error response.
4. WHERE streaming is enabled, WHEN a user submits a query against the pre-ingested Corpus, THE System SHALL deliver the first answer token to the Web_UI within 2 seconds measured from receipt of the query by the System.
5. THE end-to-end query latency budget SHALL apply only to interactive query answering and SHALL NOT apply to the offline Ingestion_Pipeline.

### Requirement 10: Evaluation and Monitoring

**User Story:** As a developer, I want retrieval and answer quality measured, so that I can demonstrate accuracy and detect regressions.

#### Acceptance Criteria

1. WHEN a recorded set of at least 30 query response times is available, THE Evaluation_Service SHALL compute the p95_Latency as the 95th percentile of that set and SHALL report it as a value in seconds with at least two decimal places.
2. IF p95_Latency is requested while fewer than 30 query response times have been recorded, THEN THE Evaluation_Service SHALL return a result indicating that the recorded sample is insufficient and SHALL NOT report a p95_Latency value.
3. WHEN an evaluation query set containing at least 10 queries, each with labeled relevant chunks, is provided, THE Evaluation_Service SHALL compute R@k over the Top_K retrieved results and MRR for that query set, each reported as a value between 0.0 and 1.0 inclusive.
4. WHEN an evaluation query set with reference answers is provided, THE Evaluation_Service SHALL compute a citation-accuracy metric, reported as a value between 0.0 and 1.0 inclusive, defined as the count of generated Citations whose cited page is among the reference answer's labeled source pages divided by the total count of generated Citations.
5. WHEN an evaluation query set with reference answers is provided, THE Evaluation_Service SHALL compute a hallucination-rate metric, reported as a value between 0.0 and 1.0 inclusive, defined as the count of generated answers containing at least one claim that cannot be matched to any retrieved chunk used for that answer divided by the total count of generated answers in the query set.
6. IF an evaluation query set is empty or omits the labels required by a requested metric, THEN THE Evaluation_Service SHALL return an error response identifying the missing labels and SHALL NOT compute that metric.
7. WHEN the recorded metrics are requested through the reporting interface, THE Evaluation_Service SHALL return the most recently computed p95_Latency, R@k, MRR, citation-accuracy, and hallucination-rate values, each annotated with the size of the query set or sample over which it was computed.

### Requirement 11: Single-Page Web UI

**User Story:** As an end user, I want a single-page chat interface that shows answers, citations, and retrieved passages, so that I can ask questions and inspect how each answer was produced.

#### Acceptance Criteria

1. THE Web_UI SHALL present a single-page chat interface, served as one page without full-page reloads between query submission and answer display, that provides a query input control for submitting queries and an answer display area for displaying answers.
2. WHEN the System returns an answer with one or more Citations, THE Web_UI SHALL display the answer text together with each associated Citation, and each displayed Citation SHALL show the source PDF filename and all associated page numbers.
3. WHEN the System returns an answer with no Citations, THE Web_UI SHALL display the answer text and SHALL display an indication that no sources are available for that answer.
4. WHEN the System returns retrieved chunks for a query, THE Web_UI SHALL display each of the returned Top_K retrieved chunks, and for each displayed chunk SHALL show its relevance score, source PDF filename, and page number.
5. WHEN the System returns an empty retrieved-chunk set for a query, THE Web_UI SHALL display an indication that no passages were retrieved.
6. THE Web_UI SHALL display a representation of the ingestion pipeline stages in the order PDF, chunking, embedding, and Vector_Store.
7. WHILE an answer is being generated and no answer or error has been received, THE Web_UI SHALL display a pending indicator, and THE Web_UI SHALL remove the pending indicator when the answer or an error is received.
8. IF the System returns an error for a query, THEN THE Web_UI SHALL remove the pending indicator and SHALL display an error message indicating that the query could not be answered, while retaining the submitted query text in the input control.

### Requirement 12: Multi-Turn Conversation

**User Story:** As an end user, I want the chatbot to remember earlier turns in my conversation, so that I can ask follow-up questions that reference what was already discussed.

#### Acceptance Criteria

1. THE System SHALL maintain a Conversation_Session that retains, in order, the prior user questions and assistant answers exchanged within a single user session.
2. WHEN a user submits a follow-up query within an existing Conversation_Session, THE System SHALL incorporate the retained Conversation_Session history into the Retrieval_Service query formulation, the Answer_Generator prompt, or both, so that the follow-up query is resolved in the context of prior turns.
3. WHEN a user submits a follow-up query that contains a pronoun or other reference to an entity introduced in an earlier turn of the same Conversation_Session, THE Answer_Generator SHALL resolve that reference using the retained Conversation_Session history.
4. THE System SHALL retain at most the most recent 10 turns and at most 4000 tokens of Conversation_Session history, and WHEN a new turn would exceed either bound AND prior Conversation_Session history exists, THE System SHALL discard the oldest turns until both bounds are satisfied.
5. WHILE the Conversation_Session contains no prior turns, WHEN a new turn is added, THE System SHALL skip the history-cleanup logic and SHALL NOT attempt to discard any turns.
6. WHEN a user starts a new conversation, THE System SHALL clear the Conversation_Session history so that no prior turn influences subsequent queries.
7. THE System SHALL scope each Conversation_Session history to its own session and SHALL NOT expose or apply one session's Conversation_Session history to any other session.
8. WHEN a user submits the first query of a Conversation_Session, THE System SHALL answer that single query correctly using only the retrieved chunks and SHALL apply no prior conversation context.

### Requirement 13: Structured Ingestion Error Log

**User Story:** As an automated maintenance agent, I want every ingestion failure recorded in a structured machine-readable log, so that I can diagnose and fix ingestion issues after a run.

#### Acceptance Criteria

1. WHEN the Ingestion_Pipeline records an extraction, OCR, embedding, or persistence failure, THE Ingestion_Pipeline SHALL write one structured, machine-readable entry to the Ingestion_Error_Log.
2. WHEN the Ingestion_Pipeline writes an entry to the Ingestion_Error_Log, THE entry SHALL identify the PDF filename, the page number where applicable, the failing stage as one of extraction, OCR, embedding, or persistence, and the failure reason.
3. THE Ingestion_Error_Log SHALL be expressed in a structured machine-readable format, with one self-contained record per failure, that an automated agent can parse without human intervention.
4. WHEN an individual extraction, OCR, embedding, or persistence failure is logged, THE Ingestion_Pipeline SHALL continue processing the remaining pages, chunks, and PDF documents without interruption.
5. WHEN the ingestion run completes, THE Ingestion_Error_Log SHALL persist to durable storage so that every recorded entry remains available for review after the run.

### Requirement 14: Docker Deployment on a CPU-Only Open-Source Stack

**User Story:** As an operator, I want the whole system deployable via Docker without a GPU, so that I can run it on commodity hardware using only open-source components.

#### Acceptance Criteria

1. THE System SHALL provide a Docker deployment for the backend, comprising the RAG API and the Qdrant Vector_Store.
2. THE System SHALL provide a Docker deployment for the frontend, comprising the single-page Web_UI.
3. THE System SHALL run the Ingestion_Pipeline, the Retrieval_Service, the Embedding_Model, and the Vector_Store using a CPU-only, open-source stack that requires no GPU and no VRAM.
4. THE System SHALL keep the LLM_Provider configurable, and WHERE the LLM_Provider is configured to a hosted provider such as Gemini, THE System SHALL route generation requests to that hosted provider while continuing to run the remaining components on the CPU-only open-source stack.
5. WHEN the backend Docker deployment starts, THE Vector_Store container SHALL load the previously persisted Corpus from durable storage and make it available to the Retrieval_Service without re-ingestion.
