# API Reference

CiteRAG exposes a REST API for document ingestion, conversation management, and querying.

**Base URL**: `http://localhost:7860`

## Authentication

Currently, no authentication is required. Treat the default configuration as local or controlled-use only; add authentication and rate limiting before exposing it publicly.

---

## Conversations

### Create Conversation

Create a new conversation session.

**Endpoint**: `POST /conversations`

**Response**: `201 Created`

```json
{
  "conversation_id": "abc123def456"
}
```

**Example**:

```bash
curl -X POST http://localhost:7860/conversations
```

---

### Submit Query

Submit a question within a conversation.

**Endpoint**: `POST /conversations/{conversation_id}/query`

**Headers**:
- `Content-Type: application/json`

**Request Body**:

```json
{
  "q": "What is the main topic of the document?",
  "top_k": 5,
  "stream": false
}
```

**Parameters**:
- `q` (string, required): The user's question
- `top_k` (integer, optional): Number of chunks to retrieve (1-100, default: 5)
- `stream` (boolean, optional): Enable Server-Sent Events streaming (default: false)

#### JSON Response (stream=false)

**Response**: `200 OK`

```json
{
  "answer": "The main topic is...",
  "citations": [
    {
      "filename": "document.pdf",
      "pages": [1, 3]
    }
  ],
  "chunks": [
    {
      "text": "Relevant passage...",
      "score": 0.89,
      "filename": "document.pdf",
      "page_number": 1,
      "chunk_position": 0,
      "language": "en"
    }
  ]
}
```

#### SSE Response (stream=true)

**Response**: `200 OK` (text/event-stream)

**Event Types**:

```
data: {"type": "token", "text": "The "}
data: {"type": "token", "text": "main "}
data: {"type": "token", "text": "topic..."}
data: {"type": "sources", "data": [{"filename": "doc.pdf", "pages": [1]}]}
data: {"type": "chunks", "data": [{...}]}
data: {"type": "end"}
```

**Example**:

```bash
# JSON response
curl -X POST http://localhost:7860/conversations/abc123/query \
  -H "Content-Type: application/json" \
  -d '{"q": "What is RAG?", "top_k": 3}'

# SSE streaming
curl -X POST http://localhost:7860/conversations/abc123/query \
  -H "Content-Type: application/json" \
  -d '{"q": "What is RAG?", "stream": true}' \
  -N
```

---

### Delete Conversation

Delete a conversation and its history.

**Endpoint**: `DELETE /conversations/{conversation_id}`

**Response**: `200 OK`

```json
{
  "message": "Conversation deleted"
}
```

**Example**:

```bash
curl -X DELETE http://localhost:7860/conversations/abc123
```

---

## Documents

### Upload Document

Upload and ingest a PDF document.

**Endpoint**: `POST /upload`

**Headers**:
- `Content-Type: multipart/form-data`

**Request Body**:
- `file`: PDF file (binary)

**Response**: `200 OK`

```json
{
  "filename": "document.pdf",
  "num_chunks": 45,
  "success": true
}
```

**Example**:

```bash
curl -X POST http://localhost:7860/upload \
  -F "file=@document.pdf"
```

---

### List Documents

List all ingested documents.

**Endpoint**: `GET /documents`

**Response**: `200 OK`

```json
{
  "documents": [
    {
      "filename": "document.pdf"
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:7860/documents
```

---

### Serve Document

Download an ingested document.

**Endpoint**: `GET /documents/{filename}`

**Response**: `200 OK` (application/pdf)

**Example**:

```bash
curl http://localhost:7860/documents/document.pdf -o document.pdf
```

---

### Delete Document

Delete an ingested document and its chunks from the vector store.

**Endpoint**: `DELETE /documents/{filename}`

**Response**: `200 OK`

```json
{
  "message": "Document deleted"
}
```

**Example**:

```bash
curl -X DELETE http://localhost:7860/documents/document.pdf
```

---

## Observability

### Health Check

Check backend and Qdrant health.

**Endpoint**: `GET /healthz`

**Response**: `200 OK`

```json
{
  "status": "ok",
  "qdrant": true,
  "corpus_count": 45
}
```

**Example**:

```bash
curl http://localhost:7860/healthz
```

---

### Metrics

Retrieve evaluation metrics.

**Endpoint**: `GET /metrics`

**Response**: `200 OK`

```json
{
  "p95_latency": {
    "value": null,
    "sample_size": 0,
    "error": null,
    "is_insufficient": true
  },
  "recall_at_k": null,
  "mrr": null,
  "citation_accuracy": null,
  "hallucination_rate": null
}
```

The example shows the initial state. A numeric p95 value requires at least 30
recorded query samples. Retrieval and answer-quality metrics are populated only
after a labeled evaluation set has been supplied to the evaluation service.

**Example**:

```bash
curl http://localhost:7860/metrics
```

---

## Error Responses

All endpoints return errors in the following format:

```json
{
  "detail": "Error message"
}
```

### Common Status Codes

- `400 Bad Request`: Invalid input
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: LLM or Qdrant unavailable
- `504 Gateway Timeout`: Request timeout

---

## Rate Limits

Currently, no rate limits are enforced. For production:

- Implement rate limiting middleware
- Use Redis for distributed rate limiting
- Set limits per IP or API key

---

## CORS

CORS is configured to allow all origins in development. For production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

---

## WebSocket Support

WebSocket support is not currently implemented. Use SSE streaming for real-time responses.

---

## Client Libraries

### Python

```python
import requests

# Create conversation
resp = requests.post("http://localhost:7860/conversations")
conv_id = resp.json()["conversation_id"]

# Submit query
resp = requests.post(
    f"http://localhost:7860/conversations/{conv_id}/query",
    json={"q": "What is RAG?", "top_k": 5}
)
print(resp.json()["answer"])
```

### JavaScript

```javascript
// Create conversation
const resp = await fetch("http://localhost:7860/conversations", {
  method: "POST"
});
const { conversation_id } = await resp.json();

// Submit a streaming query with fetch. The response body is text/event-stream.
const response = await fetch(
  `http://localhost:7860/conversations/${conversation_id}/query`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q: "What is RAG?", stream: true })
  }
);

const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  console.log(decoder.decode(value, { stream: true }));
}
```

---

## API Versioning

Currently, no API versioning is implemented. Breaking changes will be announced in the changelog.

Future versions will use URL-based versioning: `/v2/conversations`
