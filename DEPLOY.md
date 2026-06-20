# Deployment Guide

## Architecture

```
Browser
  └─► Frontend  (GitHub Pages / Cloudflare Pages — static)
        └─► Backend  (Hugging Face Spaces — Docker)
              └─► Qdrant Cloud  (free tier, persistent vector store)
              └─► Gemini API    (LLM, key in HF secrets)
```

---

## 1. Qdrant Cloud (free persistent vector store)

1. Sign up at https://cloud.qdrant.io (free tier: 1 cluster, 1 GB)
2. Create a cluster — note the **Cluster URL** and generate an **API Key**
3. You'll use these as `QDRANT_URL` and `QDRANT_API_KEY` in the backend

---

## 2. Backend on Hugging Face Spaces

The backend is a Docker Space. HF reads `backend/README.md` for the Space config.

### Steps

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Docker**
   - Hardware: CPU Basic (free)
2. Push the `backend/` folder as the Space repo (or use the existing sync workflow)
3. In Space Settings → **Variables and secrets**, add:
   - `GOOGLE_API_KEY` = your Gemini key  *(secret)*
   - `QDRANT_URL` = `https://your-cluster.cloud.qdrant.io`  *(variable)*
   - `QDRANT_API_KEY` = your Qdrant Cloud key  *(secret)*
4. The Space builds and starts automatically
5. Note your Space URL: `https://YOUR_HF_USERNAME-rag-pdf-chatbot.hf.space`

### Corpus ingestion

To pre-ingest PDFs into the Space:
- Add `CORPUS_DIR=/data/corpus` and mount PDFs — or ingest them locally first
  and point `QDRANT_URL` at the same Qdrant Cloud cluster from your local machine:

```bash
cd backend
QDRANT_URL=https://your-cluster.cloud.qdrant.io \
QDRANT_API_KEY=your-key \
CORPUS_DIR=./corpus \
python -m backend.ingestion.run
```

Once ingested into Qdrant Cloud, the HF Space just queries without re-ingesting
(`SKIP_INGESTION=true` in the Space settings speeds up cold starts).

---

## 3. Frontend on GitHub Pages (or Cloudflare Pages)

The frontend is a static Vite/React build. The only thing it needs at build time
is the backend URL.

### GitHub Pages

```bash
cd frontend

# Set the backend URL (your HF Space URL)
export VITE_API_URL=https://YOUR_HF_USERNAME-rag-pdf-chatbot.hf.space

npm ci
npm run build          # outputs to frontend/dist/

# Deploy dist/ to GitHub Pages however you prefer
# (gh-pages package, GitHub Actions, etc.)
```

Or add a GitHub Actions workflow that sets `VITE_API_URL` from a repository secret
and deploys `dist/` on every push to `main`.

### Cloudflare Pages

1. Connect the repo in Cloudflare Pages dashboard
2. Build command: `npm run build`
3. Build output: `dist`
4. Environment variable: `VITE_API_URL` = `https://YOUR_HF_USERNAME-rag-pdf-chatbot.hf.space`

---

## 4. Local development with docker compose

For local testing (no cloud accounts needed):

```bash
# Create a .env file at the repo root
cat > .env <<EOF
GOOGLE_API_KEY=your-gemini-key
VITE_API_URL=http://localhost:7860
EOF

# Drop PDFs into ./corpus/
mkdir -p corpus

# Build and start everything
docker compose up --build
```

- Frontend: http://localhost:80
- Backend API: http://localhost:7860
- Qdrant dashboard: http://localhost:6333/dashboard

---

## 5. Secrets summary

| Where | Secret | Required |
|-------|--------|----------|
| HF Space | `GOOGLE_API_KEY` | Yes |
| HF Space | `QDRANT_URL` | Yes (Qdrant Cloud URL) |
| HF Space | `QDRANT_API_KEY` | Yes (Qdrant Cloud) |
| Frontend build | `VITE_API_URL` | Yes (HF Space URL) |
| Local `.env` | `GOOGLE_API_KEY` | Yes (local only) |
