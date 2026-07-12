# Frequently Asked Questions (FAQ)

## General

### What is CiteRAG?

CiteRAG is a Retrieval-Augmented Generation (RAG) system that allows you to chat with your PDF documents. It provides answers grounded in the documents with page-level citations, so you can verify every claim.

### How is this different from other RAG systems?

Key differentiators:
- **Page-level citations**: Every answer includes `[filename, page X]` references
- **Hybrid retrieval**: Combines dense (semantic) and sparse (lexical) search with RRF fusion
- **Multi-turn conversations**: Maintains context across multiple questions
- **CPU-only**: No GPU required, runs on commodity hardware
- **Multilingual embeddings**: Uses BGE-M3 for document and query embeddings

### Is it free to use?

Yes, the software is open-source (MIT License). You still need a Google API key for the current Gemini provider, and Google's pricing or quotas can change.

---

## Setup & Installation

### What are the system requirements?

**Minimum**:
- 2 CPU cores
- 8 GB RAM
- 10 GB disk space
- Docker installed

**Recommended**:
- 4+ CPU cores
- 16 GB RAM
- 20 GB disk space (for model cache)

### Do I need a GPU?

No GPU is required by the default configuration. Embedding and retrieval run on the CPU; actual resource use depends on the documents and model runtime.

### How do I get a Google API key?

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API key"
4. Copy the key and add it to your `.env` file

Google's quotas and pricing are account- and time-dependent; check Google's current documentation for details.

### Can I use a different LLM provider?

The code has a provider interface and registry, but only Gemini is registered in the current repository. To add another provider:

1. Create a new file in `backend/providers/` (e.g., `openai.py`)
2. Implement the `LLMProvider` interface
3. Update `backend/providers/__init__.py`
4. Set `LLM_PROVIDER=openai` in your `.env`

See the Gemini provider as a reference implementation.

---

## Usage

### What file formats are supported?

Currently, only PDF files are supported. Future versions may add:
- DOCX
- TXT
- Markdown
- HTML
- EPUB

### How many PDFs can I upload?

There is no published capacity benchmark for this repository. Embedded Qdrant is the default local-development mode; larger deployments should be measured on the target hardware and may use a separately managed Qdrant instance.

### How long does ingestion take?

It depends on PDF length, whether OCR is needed, embedding model load time, and the host machine. Start with a small document when checking a new setup.

### Can I delete documents after uploading?

Yes, use the `DELETE /documents/{filename}` endpoint. This removes:
- The document file
- All chunks from the vector store
- All associated metadata

### How accurate are the citations?

Citation accuracy depends on:
- PDF quality (digital vs scanned)
- Chunking boundaries
- LLM hallucination rate

The repository includes citation and evaluation code, but it does not publish a benchmark result. Verify important claims by checking the cited source pages.

### Can I chat in languages other than English?

The embedding model is multilingual and the application detects the query language to guide the generated response. Actual answer quality depends on the source document and the LLM provider.

---

### Why is the first query slow?

The first query may load the embedding model into memory. Subsequent queries can behave differently depending on cache state and host resources.

### Can I speed up generation?

Yes:
1. **Disable reranking**: Set `RERANK_ENABLED=false`
2. **Reduce top_k**: Lower from 5 to 3 (fewer passages are retrieved)
3. **Use streaming**: Perceive faster responses via SSE streaming
4. **Choose an appropriate Gemini model** through the supported configuration for your account

### Can I expose it publicly?

The default app has no authentication or rate limiting and allows permissive CORS for development. Treat it as a local or controlled deployment until those controls are configured. See [DEPLOYMENT.md](DEPLOYMENT.md) and [SECURITY.md](../SECURITY.md).

---

## Troubleshooting

### "GOOGLE_API_KEY environment variable not set"

Solution:
1. Copy `.env.example` to `.env`
2. Add your Google API key: `GOOGLE_API_KEY=your-key-here`
3. Restart the backend: `docker compose restart backend`

### "Embedding model download fails"

This usually happens due to network timeouts. Solutions:
1. Increase timeout: `HF_HUB_DOWNLOAD_TIMEOUT=300` in `.env`
2. Check firewall/proxy settings
3. Manually download the model:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
   ```

### "Out of memory" error

Solutions:
1. Reduce concurrent requests
2. Lower `DEFAULT_TOP_K` (default: 5 → 3)
3. Disable reranking: `RERANK_ENABLED=false`
4. Use Qdrant Cloud instead of embedded mode
5. Allocate more RAM (16 GB recommended)

### "Session expired" or "Session not found"

Sessions expire after 15 minutes of inactivity. This is by design to prevent memory leaks. Solutions:
1. Refresh the session: `POST /sessions/{id}/refresh`
2. Adjust timeout in `backend/app.py`: `SESSION_TIMEOUT_MINUTES`
3. Use a shared session store if you deploy multiple backend instances

### "OCR not working" or "Tesseract not found"

Tesseract is included in the Docker image. For local development:

**Ubuntu/Debian**:
```bash
sudo apt-get install tesseract-ocr
```

**macOS**:
```bash
brew install tesseract
```

**Windows**:
Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### "CORS error" in frontend

The backend allows all origins by default. If you see CORS errors:
1. Check the backend is running: `curl http://localhost:7860/healthz`
2. Verify `VITE_API_URL` in frontend `.env`
3. Check browser console for exact error

---

## Advanced

### How do I change chunk size?

Chunk size affects:
- **Larger chunks**: More context, but less precise retrieval
- **Smaller chunks**: More precise, but may miss context

To change:
```bash
CHUNK_MAX_TOKENS=800  # Default
CHUNK_OVERLAP_TOKENS=150  # Default
```

Test different values based on your document structure.

### Can I self-host the embedding model?

Yes! BGE-M3 is downloaded from Hugging Face and cached locally. The model runs on your hardware (no external API calls for embeddings).

### What is Reciprocal Rank Fusion (RRF)?

RRF combines dense (semantic) and sparse (lexical) search results:
1. Dense search finds semantically similar chunks
2. Sparse search finds keyword matches
3. RRF merges the two lists, boosting items that appear in both

It is intended to combine complementary retrieval signals; this repository does not claim a measured improvement without a labeled evaluation set.

### How do I add custom metadata to chunks?

Modify `backend/ingestion/chunker.py` to add metadata:
```python
chunk_metadata = {
    "text": chunk_text,
    "filename": filename,
    "page_number": page_num,
    "custom_field": "custom_value"  # Add here
}
```

Then filter during retrieval using Qdrant's metadata filtering.

### Can I use this for non-PDF documents?

Yes, but you'll need to implement a new loader:
1. Create a loader in `backend/utils/loaders.py`
2. Handle extraction for your format
3. Return plain text with metadata
4. The rest of the pipeline remains the same

---

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines. Quick start:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a PR

We welcome:
- Bug fixes
- New features
- Documentation improvements
- Test coverage
- Performance optimizations

### I found a security vulnerability. What should I do?

**Do NOT open a public issue.** Instead:
1. Email the maintainer directly, or
2. Use GitHub's private vulnerability reporting

See [SECURITY.md](../SECURITY.md) for details.

---

## Licensing

### Can I use this commercially?

Yes! CiteRAG is MIT Licensed, which allows:
- Commercial use
- Modification
- Distribution
- Private use

Just include the original license and copyright notice.

### What about the dependencies?

All major dependencies are permissively licensed:
- **FastAPI**: MIT
- **BGE-M3**: MIT
- **Qdrant**: Apache 2.0
- **React**: MIT

Always review licenses before deploying to production.

---

## Still have questions?

- 📖 Check the [Documentation](.)
- 🐛 [Open an issue](https://github.com/melbinjp/citerag/issues)
