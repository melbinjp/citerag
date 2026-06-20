AI/ML CHALLENGES
Challenge 1:RAG Chatbot 
Objective
 Build a Retrieval-Augmented Generation (RAG) chatbot that answers user queries from a large private corpus of PDFs (≥10 PDFs, each ≥200 pages) using free / open-source embedding models and free / open-source vector DB, with an end-to-end response latency of 2–5 seconds.
Input sources / Dataset
Multiple PDFs (minimum 10 PDFs, each ≥200 pages).
PDFs may contain text, images, scanned pages. Must support:
* native text extraction (where available)
* OCR for scanned pages/embedded images
Core AI Tasks
1. Ingestion & Preprocessing
* Extract text from PDFs (pdfminer / PyMuPDF / pypdf for native).
* OCR scanned pages (Tesseract or other OSS OCR).
* Clean, normalize, remove headers/footers, detect language.

2. Chunking & Metadata
   * Chunk text into passages (e.g., 500–1000 tokens) with overlap (10–30%).
   * Keep metadata per chunk: PDF id, filename, page number, bounding box (if relevant).
3. Embedding
   * Use a free / open-source embedding model to embed chunks.
   * Persist embeddings to vector DB.
4. Indexing & Retrieval
   * Build an ANN index (HNSW / IVF+PQ) for fast nearest-neighbor search.
   * Return top-K candidate chunks with metadata & scores.
5. Reranking / Filtering
   * Optional lightweight reranker (cross-encoder or lexical filter) to refine top results.

6. Generation / Answering (RAG)
      * Use an LLM (open-source or hosted) to synthesize answers conditioned on retrieved chunks, include provenance (PDF filename + page numbers), and follow safety/citation instructions.
7. Latency & Throughput
      * Ensure query -> answer time within 2–5s for typical queries.
8. Evaluation & Monitoring
      * Measure latency (p95), relevance (R@k, MRR), hallucination rate, citation accuracy.


Non-functional requirements
      * Open / Free: Embedding model and vector DB must be free/open-source.
      * Scalability: Support ingestion of many large PDFs (>200 pages each).
      * Latency: 2–5 seconds (notice: depends on hardware; see tuning below).
      * Explainability: Provide sources (PDF name + page) for each answer.
      * Reproducibility: Precompute embeddings; deterministic chunking rules.

Deliverable
A live demo RAG chatbot system where:
         * The system has ingested multiple PDFs (minimum 10 PDFs, each ≥200 pages) into a vector database using open-source embeddings.

         * A user asks questions through a chat interface (web UI or API).

         * The system retrieves the most relevant content from the PDFs in real time using the RAG pipeline.

         * The chatbot generates accurate answers within 2–5 seconds latency.

         * Each answer includes source references (PDF name + page number).

         * The demo also shows:

            * Document ingestion pipeline (PDF → chunking → embedding → vector DB).

            * Retrieval visualization (top retrieved chunks).

            * Final generated answer with citations.

Output:
 Real-time question answering from a large PDF knowledge base with fast response, traceable sources, and open-source stack.
Challenge 2: AI-Driven Automated Interviewer for Project Presentations
Objective
Build an AI system that listens to a student presenting a project (screen share + speech) and conducts an adaptive interview based on content and responses.
Important Constraint
               * The company will NOT provide any API key.
               * The entire system must run fully client-side.
               * The AI model must run locally in the browser.
               * OCR must be client-side.
               * Speech-to-Text must be client-side.
Functional Requirements
               * Presentation Understanding
               * Extract content from screens using client-side OCR.
               * Transcribe student speech using client-side STT.
               * Analyze UI, code snippets, slides, diagrams — all locally in the browser.


Dynamic Interviewing
               * Generate context-aware questions from extracted content using a client-side model.
               * Ask follow-up questions based on responses and screen content.
               * All logic must execute locally without external APIs.


Evaluation & Feedback
Score student on:
               * Technical depth
               * Clarity of explanation
               * Originality
               * Understanding of implementation
Generate a structured feedback report locally without server-side processing.
Deliverable
A live demo where a student presents a project, the system interviews them in real time, and produces a score + feedback report — fully client-side without using any external API keys.