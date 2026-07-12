# Deployment Guide

This guide covers the repository's local Docker path and the configuration
points to review for a hosted deployment. Provider-specific cloud deployments
are not packaged or benchmarked in this repository.

## Docker Deployment (Recommended)

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/melbinjp/citerag.git
cd citerag

# 2. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 3. Start backend with Docker Compose
docker compose up --build

# 4. Start frontend (in a separate terminal)
cd frontend
npm install
npm run dev
```

Access the application at http://localhost:5173

### Production Deployment

For production, build the frontend and serve it via nginx:

```bash
# Build frontend
cd frontend
npm install
npm run build

# The dist/ folder contains the production build
```

Update `docker-compose.yml` to add an nginx service or use a reverse proxy.

## Qdrant Cloud Deployment

For production workloads, use Qdrant Cloud instead of embedded mode:

1. Create a Qdrant Cloud cluster at https://cloud.qdrant.io
2. Update `.env`:

```bash
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

3. Deploy as usual. The backend will use the cloud cluster.

## Environment Variables

### Required

- `GOOGLE_API_KEY` - Google Gemini API key

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | LLM provider to use |
| `QDRANT_URL` | (empty) | Qdrant Cloud URL (leave empty for embedded) |
| `QDRANT_API_KEY` | (empty) | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | `corpus` | Collection name in Qdrant |
| `SKIP_INGESTION` | `false` | Skip startup corpus ingestion |
| `RERANK_ENABLED` | `false` | Enable reranking |
| `CHUNK_MAX_TOKENS` | `800` | Maximum tokens per chunk |
| `DEFAULT_TOP_K` | `5` | Default retrieval results |
| `GENERATION_TIMEOUT_S` | `30` | LLM generation timeout |

## Cloud platform options

The following platforms can run a Docker-based service, but each deployment
still needs its own secrets, persistent storage, HTTPS, authentication, rate
limiting, and health-check configuration. Treat these as starting points rather
than tested one-click deployments.

### Hugging Face Spaces

1. Create a new Docker Space
2. Push the repository
3. Add `GOOGLE_API_KEY` in Space settings
4. Select hardware after measuring the embedding and OCR workload

### Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Add environment variables in the Railway dashboard.

### Render

1. Create a new Web Service
2. Connect your GitHub repository
3. Set build command: `docker build -t citerag ./backend`
4. Set start command: `docker run -p 7860:7860 citerag`
5. Add environment variables

### DigitalOcean App Platform

1. Create a new app from GitHub
2. Select the repository
3. Configure as a Docker app
4. Add environment variables
5. Deploy

## Deployment considerations

### Backend

- **CPU and memory**: Measure embedding and OCR load on the target machine.
- **Storage**: Persist `/data` for embedded Qdrant, ingestion logs, and the
  model cache.

### Frontend

- Serve the production build via CDN (Cloudflare, Vercel, Netlify).
- Use nginx or Caddy as a reverse proxy.

### Vector Store

- For larger corpora, evaluate Qdrant Cloud or a dedicated Qdrant server on
  the target workload.
- Consider Qdrant quantization only after measuring its effect on retrieval.

## SSL/TLS

Use a reverse proxy (nginx, Caddy, Traefik) to handle HTTPS:

**nginx example:**

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location /api/ {
        proxy_pass http://localhost:7860/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location / {
        root /path/to/frontend/dist;
        try_files $uri $uri/ /index.html;
    }
}
```

## Monitoring

- **Health Check**: `GET /healthz`
- **Metrics**: `GET /metrics`
- **Logs**: Check Docker logs with `docker compose logs -f backend`

## Backup

Backup the Qdrant data directory:

```bash
# With embedded Qdrant
docker compose down
tar -czf qdrant-backup.tar.gz backend_data/
docker compose up -d
```

## Troubleshooting

### Backend won't start

- Check `GOOGLE_API_KEY` is set correctly
- Ensure port 7860 is available
- Check logs: `docker compose logs backend`

### Embedding model download fails

- Increase `HF_HUB_DOWNLOAD_TIMEOUT` (default: 120s)
- Check network connectivity to Hugging Face Hub
- Manually download model and mount as volume

### Out of memory

- Reduce `DEFAULT_TOP_K`
- Disable reranking (`RERANK_ENABLED=false`)
- Use Qdrant Cloud instead of embedded mode

### Slow queries

- Enable reranking for better precision
- Reduce `DEFAULT_TOP_K`
- Check Qdrant Cloud cluster tier
- Add a shared cache or session store only if the deployment requires it.

## Security Hardening

See [SECURITY.md](../SECURITY.md) for detailed security recommendations.

Quick checklist:
- [ ] Use HTTPS in production
- [ ] Restrict CORS origins
- [ ] Add rate limiting
- [ ] Implement authentication
- [ ] Rotate API keys regularly
- [ ] Keep dependencies updated
- [ ] Monitor logs for anomalies
