import React from 'react';
import './RetrievedChunks.css';

/**
 * RetrievedChunks — displays the Top_K retrieved passages returned by the
 * Retrieval_Service, or an empty-state message when none were retrieved.
 *
 * Props:
 *   chunks {Array} — array of retrieved chunk objects with shape:
 *     {
 *       text: string,
 *       score: number,
 *       metadata: {
 *         filename: string,
 *         page_number: number,
 *         chunk_position: number,
 *         language: string
 *       }
 *     }
 *
 * Requirements: 11.4, 11.5
 */
const SNIPPET_MAX_CHARS = 200;

function truncate(text, maxChars) {
  if (!text) return '';
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars).trimEnd() + '…';
}

function formatScore(score) {
  if (typeof score !== 'number' || isNaN(score)) return '—';
  // RRF scores are typically small floats (e.g. 0.033); display as a 4-decimal
  // float but also show a percentage hint when the value is clearly in [0, 1].
  if (score <= 1) {
    return `${(score * 100).toFixed(1)}%`;
  }
  return score.toFixed(4);
}

const RetrievedChunks = ({ chunks }) => {
  const hasChunks = Array.isArray(chunks) && chunks.length > 0;

  return (
    <section
      className="retrieved-chunks"
      aria-label="Retrieved passages"
    >
      <h3 className="retrieved-chunks__heading">
        Retrieved Passages
        {hasChunks && (
          <span className="retrieved-chunks__count" aria-live="polite">
            {chunks.length}
          </span>
        )}
      </h3>

      {!hasChunks ? (
        /* Requirement 11.5 — empty state */
        <p
          className="retrieved-chunks__empty"
          role="status"
          aria-live="polite"
        >
          No passages were retrieved for this query.
        </p>
      ) : (
        /* Requirement 11.4 — render each Top_K chunk */
        <ol
          className="retrieved-chunks__list"
          aria-label={`${chunks.length} retrieved passage${chunks.length !== 1 ? 's' : ''}`}
        >
          {chunks.map((chunk, index) => {
            const meta = chunk.metadata || {};
            const filename = meta.filename || 'Unknown file';
            const pageNumber = meta.page_number != null ? meta.page_number : '—';
            const score = chunk.score;
            const snippetText = truncate(chunk.text, SNIPPET_MAX_CHARS);

            return (
              <li
                key={`${filename}-${pageNumber}-${index}`}
                className="retrieved-chunks__item"
                aria-label={`Passage ${index + 1}: ${filename}, page ${pageNumber}`}
              >
                {/* Rank badge */}
                <span className="retrieved-chunks__rank" aria-hidden="true">
                  #{index + 1}
                </span>

                {/* Score — Requirement 11.4 */}
                <div className="retrieved-chunks__score">
                  <span className="retrieved-chunks__label">Score</span>
                  <span
                    className="retrieved-chunks__score-value"
                    title={`Raw score: ${score}`}
                  >
                    {formatScore(score)}
                  </span>
                </div>

                {/* Source filename — Requirement 11.4 */}
                <div className="retrieved-chunks__source">
                  <span className="retrieved-chunks__label" aria-hidden="true">Source</span>
                  <span
                    className="retrieved-chunks__filename"
                    title={filename}
                    aria-label={`Source file: ${filename}`}
                  >
                    {filename}
                  </span>
                </div>

                {/* Page number — Requirement 11.4 */}
                <div className="retrieved-chunks__page">
                  <span className="retrieved-chunks__label" aria-hidden="true">Page</span>
                  <span
                    className="retrieved-chunks__page-value"
                    aria-label={`Page ${pageNumber}`}
                  >
                    {pageNumber}
                  </span>
                </div>

                {/* Chunk text snippet */}
                {snippetText && (
                  <blockquote className="retrieved-chunks__snippet">
                    {snippetText}
                  </blockquote>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
};

export default RetrievedChunks;
