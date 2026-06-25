import axios from 'axios';

// Configurable via VITE_API_URL env var; defaults to the local backend so the
// app works out-of-the-box in local development.
const API_ROOT = import.meta.env.VITE_API_URL || 'http://localhost:7860';

const api = axios.create({
  baseURL: API_ROOT,
});

export const createSession = async () => {
  const response = await api.post('/sessions');
  return response.data;
};

// Add a response interceptor
api.interceptors.response.use(
  (response) => response, // Simply return a successful response
  async (error) => {
    const originalRequest = error.config;
    const status = error.response ? error.response.status : null;
    const isSessionEndpoint = originalRequest.url.includes('/sessions/');
    const isCreatingSession = originalRequest.url.endsWith('/sessions');

    // Check if it's a 404 on a session endpoint, but not on the /sessions create endpoint itself
    // And also check a flag to prevent infinite loops if session creation fails repeatedly.
    if (status === 404 && isSessionEndpoint && !isCreatingSession && !originalRequest._retry) {
      console.log('Session expired or not found. Attempting to create a new one.');
      originalRequest._retry = true; // Mark that we've tried to refresh the session

      try {
        // Step 1: Create a new session
        const newSession = await createSession();
        const newSessionId = newSession.session_id;
        console.log('New session created:', newSessionId);

        // Step 2: Update localStorage
        localStorage.setItem('sessionId', newSessionId);

        // Step 3: Dispatch a custom event to notify the app of the new session
        window.dispatchEvent(new CustomEvent('session-updated'));

        // Step 4: Update the original request's URL with the new session ID
        const oldSessionId = originalRequest.url.split('/')[2];
        originalRequest.url = originalRequest.url.replace(oldSessionId, newSessionId);

        // Step 5: Retry the original request
        console.log('Retrying original request with new session ID:', originalRequest.url);
        return api(originalRequest);
      } catch (e) {
        console.error('Failed to create a new session or retry the request.', e);
        // If creating a new session fails, we should probably inform the user.
        // For now, we'll just reject the promise.
        return Promise.reject(e);
      }
    }

    // For all other errors, just pass them on
    return Promise.reject(error);
  }
);


export const ingestFile = async (sessionId, file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post(`/sessions/${sessionId}/ingest`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const ingestUrl = async (sessionId, url) => {
  const response = await api.post(`/sessions/${sessionId}/ingest`, { url }, {
    headers: {
      'Content-Type': 'application/json',
    },
  });
  return response.data;
};

export const query = async (sessionId, q, doc_ids = null) => {
    const payload = { q };
    if (doc_ids) {
        payload.doc_ids = doc_ids;
    }
    const response = await api.post(`/sessions/${sessionId}/query`, payload);
    if (response.data.error) {
        throw new Error(response.data.error);
    }
    return response.data;
};

export const queryStream = async function*(sessionId, q, doc_ids = null) {
    const payload = { q, stream: true };
    if (doc_ids) {
        payload.doc_ids = doc_ids;
    }
    
    const response = await fetch(`${API_ROOT}/sessions/${sessionId}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        let errMsg = `HTTP error! status: ${response.status}`;
        try {
            const errData = await response.json();
            if (errData.detail) errMsg = errData.detail;
        } catch (e) {}
        throw new Error(errMsg);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const dataStr = line.substring(6);
                try {
                    const data = JSON.parse(dataStr);
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    yield data;
                } catch (e) {
                    if (e.message !== 'Unexpected end of JSON input') {
                        throw e;
                    }
                }
            }
        }
    }
};

export const deleteDocument = async (sessionId, docId) => {
    const response = await api.delete(`/sessions/${sessionId}/documents/${docId}`);
    return response.data;
};

export const getSessionStatus = async (sessionId) => {
    const response = await api.get(`/sessions/${sessionId}/status`);
    return response.data;
};

export const refreshSession = async (sessionId) => {
    const response = await api.post(`/sessions/${sessionId}/refresh`);
    return response.data;
};

export const checkSessionHealth = async (sessionId) => {
    const response = await api.get(`/sessions/${sessionId}/health`);
    return response.data;
};

// ---------------------------------------------------------------------------
// Conversation API (new RAG backend: POST /conversations, etc.)
// ---------------------------------------------------------------------------

/**
 * Create a new conversation session.
 * POST /conversations → { conversation_id: string }
 */
export const createConversation = async () => {
  const response = await api.post('/conversations');
  return response.data; // { conversation_id }
};

/**
 * Delete a conversation session and clear its history.
 * DELETE /conversations/{id} → 200
 */
export const deleteConversation = async (conversationId) => {
  const response = await api.delete(`/conversations/${conversationId}`);
  return response.data;
};

/**
 * Fetch the latest evaluation metrics.
 * GET /metrics → metrics object
 */
export const getMetrics = async () => {
  const response = await api.get('/metrics');
  return response.data;
};

/**
 * Fetch the service health status.
 * GET /healthz → health status
 */
export const getHealth = async () => {
  const response = await api.get('/healthz');
  return response.data;
};

/**
 * Stream a query against a conversation session.
 *
 * POST /conversations/{id}/query  body: { q, top_k?, stream: true }
 *
 * Yields typed events consumed from the SSE stream:
 *   { type: 'token',   text: string }   — an answer token
 *   { type: 'sources', data: [...] }    — citation objects (Req 11.2)
 *   { type: 'chunks',  data: [...] }    — retrieved chunks  (Req 11.4)
 *   { type: 'end' }                     — stream finished   (Req 11.7)
 *
 * Throws an Error on `error` events (Req 11.7, 11.8).
 *
 * @param {string} conversationId
 * @param {string} q
 * @param {{ top_k?: number }} [options]
 */
export const queryConversationStream = async function* (conversationId, q, options = {}) {
  const payload = { q, stream: true };
  if (options.top_k != null) {
    payload.top_k = options.top_k;
  }

  const response = await fetch(`${API_ROOT}/conversations/${conversationId}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errMsg = `HTTP error! status: ${response.status}`;
    try {
      const errData = await response.json();
      if (errData.detail) errMsg = errData.detail;
      else if (errData.message) errMsg = errData.message;
    } catch (_) {
      // ignore JSON parse failure — keep the HTTP status message
    }
    throw new Error(errMsg);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE events are delimited by double newlines
    const parts = buffer.split('\n\n');
    buffer = parts.pop(); // keep the incomplete trailing part

    for (const part of parts) {
      // An SSE event block may have multiple lines; find the `data:` line
      const dataLine = part
        .split('\n')
        .find((l) => l.startsWith('data: '));

      if (!dataLine) continue;

      const dataStr = dataLine.substring(6).trim();
      if (!dataStr) continue;

      let event;
      try {
        event = JSON.parse(dataStr);
      } catch (_) {
        // Malformed JSON — skip this event
        continue;
      }

      const { type } = event;

      if (type === 'token') {
        // Req 11.7: answer tokens stream in before `end`
        yield { type: 'token', text: event.text ?? '' };
      } else if (type === 'sources') {
        // Req 11.2: citation list arrives as a sources event
        yield { type: 'sources', data: event.data ?? [] };
      } else if (type === 'chunks') {
        // Req 11.4: retrieved chunks arrive as a chunks event
        yield { type: 'chunks', data: event.data ?? [] };
      } else if (type === 'end') {
        // Req 11.7: pending indicator should be removed
        yield { type: 'end' };
        return; // generator is done
      } else if (type === 'error') {
        // Req 11.7, 11.8: throw so the caller can show the error and
        // preserve the input text
        throw new Error(event.message ?? 'Unknown streaming error');
      }
      // Unknown event types are silently ignored for forward compatibility
    }
  }
};

/**
 * Upload a PDF file to the global Qdrant corpus.
 * POST /upload → { filename, num_chunks, success }
 */
export const uploadGlobalFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

/**
 * Get all unique documents in the global Qdrant collection.
 * GET /documents → { documents: [{ filename }] }
 */
export const getGlobalDocuments = async () => {
  const response = await api.get('/documents');
  return response.data;
};

/**
 * Delete a document from the global Qdrant collection.
 * DELETE /documents/{filename} → 200
 */
export const deleteGlobalDocument = async (filename) => {
  const response = await api.delete(`/documents/${encodeURIComponent(filename)}`);
  return response.data;
};

export default api;
