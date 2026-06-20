import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import CitationList from './components/CitationList';
import RetrievedChunks from './components/RetrievedChunks';
import PipelineStages from './components/PipelineStages';
import ThemeSwitcher from './components/ThemeSwitcher';
import { queryConversationStream, createConversation } from './services/api';

/**
 * Single-page chat interface for the RAG PDF Chatbot.
 *
 * Requirements:
 *   11.1 — One page, no full-page reloads, query input + answer display area
 *   11.7 — Pending indicator shown during generation, removed on answer/error
 *   11.8 — On error: remove pending indicator, show "could not be answered"
 *           message, RETAIN the submitted query text in the input control
 */
function App() {
  // --- State ---
  const [inputText, setInputText] = useState('');         // controlled input; retained on error (Req 11.8)
  const [submittedQuery, setSubmittedQuery] = useState(''); // the query that was submitted
  const [answer, setAnswer] = useState('');               // accumulated answer tokens
  const [citations, setCitations] = useState([]);         // citation objects { filename, pages }
  const [chunks, setChunks] = useState([]);               // retrieved chunk objects
  const [isPending, setIsPending] = useState(false);      // Req 11.7
  const [error, setError] = useState(null);               // Req 11.8
  const [conversationId, setConversationId] = useState(null);

  // Keep a ref to the current stream so we can ignore stale results if the
  // component is unmounted while a stream is in flight.
  const activeStreamRef = useRef(0);

  // --- Create a conversation session on mount ---
  useEffect(() => {
    let cancelled = false;
    createConversation()
      .then((data) => {
        if (!cancelled) {
          setConversationId(data.conversation_id);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          console.error('Failed to create conversation:', err);
          // Still allow the user to try submitting; handleSubmit will surface
          // the error via the error state.
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // --- Submit handler ---
  const handleSubmit = async (e) => {
    e.preventDefault();
    const query = inputText.trim();
    if (!query || isPending) return;

    // Increment generation counter so stale callbacks are ignored
    const generation = ++activeStreamRef.current;

    setSubmittedQuery(query);
    setIsPending(true);     // Req 11.7 — show pending indicator
    setAnswer('');
    setCitations([]);
    setChunks([]);
    setError(null);

    try {
      const convId = conversationId;
      if (!convId) {
        throw new Error('Conversation session is not ready. Please wait a moment and try again.');
      }

      let accAnswer = '';

      for await (const event of queryConversationStream(convId, query)) {
        // Discard events from a superseded submission
        if (generation !== activeStreamRef.current) return;

        if (event.type === 'token') {
          accAnswer += event.text;
          setAnswer(accAnswer);
        } else if (event.type === 'sources') {
          setCitations(event.data ?? []);
        } else if (event.type === 'chunks') {
          setChunks(event.data ?? []);
        } else if (event.type === 'end') {
          // Req 11.7 — remove pending indicator when answer received
          setIsPending(false);
        }
      }

      // Ensure pending is cleared even if 'end' event was not yielded
      if (generation === activeStreamRef.current) {
        setIsPending(false);
      }
    } catch (err) {
      if (generation !== activeStreamRef.current) return;
      // Req 11.7 — remove pending indicator on error
      // Req 11.8 — display error message; inputText is NOT cleared (retained)
      setIsPending(false);
      setError(err.message || 'The query could not be answered. Please try again.');
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift) for quick queries
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const hasAnswer = answer.length > 0;
  const hasResult = hasAnswer || citations.length > 0 || chunks.length > 0;

  return (
    // Req 11.1 — single page, no full-page reloads
    <div className="container">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-content">
          <div>
            <h1>📄 RAG PDF Chatbot</h1>
            <p>Ask questions about your document corpus — answers with page-level citations</p>
          </div>
          <div className="header-controls">
            <ThemeSwitcher />
          </div>
        </div>
      </header>

      <main>
        {/* ── Pipeline stages (Req 11.6) ── */}
        <PipelineStages />

        {/* ── Chat Panel (Req 11.1) ── */}
        <section className="chat-panel" aria-label="Chat interface">
          {/* Query input (Req 11.1) */}
          <form
            className="chat-form"
            onSubmit={handleSubmit}
            aria-label="Submit a question"
          >
            <label htmlFor="query-input" className="chat-label">
              Ask a question
            </label>
            <textarea
              id="query-input"
              className="chat-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your question and press Enter or click Ask…"
              rows={3}
              disabled={isPending}
              aria-label="Query input"
              aria-describedby={error ? 'chat-error' : undefined}
            />
            <button
              type="submit"
              className="chat-submit-btn"
              disabled={isPending || !inputText.trim()}
              aria-busy={isPending}
            >
              {isPending ? 'Generating…' : 'Ask'}
            </button>
          </form>

          {/* Req 11.7 — pending indicator, shown only while generating */}
          {isPending && (
            <div
              className="chat-pending"
              role="status"
              aria-live="polite"
              aria-label="Generating answer"
            >
              <span className="chat-spinner" aria-hidden="true" />
              <span>Generating answer…</span>
            </div>
          )}

          {/* Req 11.8 — error state: show message, inputText is retained above */}
          {error && !isPending && (
            <div
              id="chat-error"
              className="chat-error"
              role="alert"
              aria-live="assertive"
            >
              <span className="chat-error-icon" aria-hidden="true">⚠️</span>
              <span>
                The query &ldquo;{submittedQuery}&rdquo; could not be answered. {error}
              </span>
            </div>
          )}

          {/* Answer display area (Req 11.1) */}
          {hasAnswer && !isPending && !error && (
            <section className="chat-answer" aria-label="Answer">
              <h2 className="chat-answer-heading">Answer</h2>
              <div className="chat-answer-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {answer}
                </ReactMarkdown>
              </div>
            </section>
          )}

          {/* Streaming answer (shown while still pending so tokens appear live) */}
          {hasAnswer && isPending && (
            <section className="chat-answer chat-answer--streaming" aria-label="Answer (streaming)">
              <h2 className="chat-answer-heading">Answer</h2>
              <div className="chat-answer-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {answer}
                </ReactMarkdown>
              </div>
            </section>
          )}
        </section>

        {/* ── Citations (Req 11.2, 11.3) — rendered after a result is available ── */}
        {hasResult && !isPending && !error && (
          <section className="chat-citations" aria-label="Citations">
            <h2 className="chat-section-heading">Sources</h2>
            <CitationList citations={citations} />
          </section>
        )}

        {/* ── Retrieved chunks (Req 11.4, 11.5) ── */}
        {hasResult && !isPending && !error && (
          <section className="chat-chunks" aria-label="Retrieved passages">
            <RetrievedChunks chunks={chunks} />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
