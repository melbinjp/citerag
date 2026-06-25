import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import CitationList from './CitationList';
import RetrievedChunks from './RetrievedChunks';
import PipelineStages from './PipelineStages';
import { queryConversationStream, createConversation } from '../services/api';

/**
 * Chat component for querying the pre-ingested persistent document corpus.
 */
function GlobalChat() {
  const [inputText, setInputText] = useState('');
  const [submittedQuery, setSubmittedQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [citations, setCitations] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState(null);
  const [conversationId, setConversationId] = useState(null);

  const activeStreamRef = useRef(0);

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
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const query = inputText.trim();
    if (!query || isPending) return;

    const generation = ++activeStreamRef.current;

    setSubmittedQuery(query);
    setIsPending(true);
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
        if (generation !== activeStreamRef.current) return;

        if (event.type === 'token') {
          accAnswer += event.text;
          setAnswer(accAnswer);
        } else if (event.type === 'sources') {
          setCitations(event.data ?? []);
        } else if (event.type === 'chunks') {
          setChunks(event.data ?? []);
        } else if (event.type === 'end') {
          setIsPending(false);
        }
      }

      if (generation === activeStreamRef.current) {
        setIsPending(false);
      }
    } catch (err) {
      if (generation !== activeStreamRef.current) return;
      setIsPending(false);
      setError(err.message || 'The query could not be answered. Please try again.');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const hasAnswer = answer.length > 0;
  const hasResult = hasAnswer || citations.length > 0 || chunks.length > 0;

  return (
    <div>
      {/* Pipeline stages */}
      <PipelineStages />

      {/* Chat Panel */}
      <section className="chat-panel" aria-label="Chat interface">
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

      {hasResult && !isPending && !error && (
        <section className="chat-citations" aria-label="Citations">
          <h2 className="chat-section-heading">Sources</h2>
          <CitationList citations={citations} />
        </section>
      )}

      {hasResult && !isPending && !error && (
        <section className="chat-chunks" aria-label="Retrieved passages">
          <RetrievedChunks chunks={chunks} />
        </section>
      )}
    </div>
  );
}

export default GlobalChat;
