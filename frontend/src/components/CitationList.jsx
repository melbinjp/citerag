import React, { useState } from 'react';
import './CitationList.css';

// Configurable API URL for document downloads.
const API_ROOT = import.meta.env.VITE_API_URL || 'http://localhost:7860';

const isUrl = (value) => /^https?:\/\//i.test(value || '');

/**
 * Resolve the link + display label for a citation. A citation that is itself a
 * URL links directly out; a filename links to the served document endpoint.
 */
function resolveCitation(citation) {
  const name = citation.filename || '';
  const url = isUrl(name);
  const href = url
    ? name
    : `${API_ROOT}/documents/${encodeURIComponent(name)}`;

  let label = name;
  if (url) {
    try {
      label = new URL(name).hostname.replace(/^www\./, '');
    } catch {
      label = name;
    }
  }

  const pageText =
    citation.pages && citation.pages.length > 0
      ? ` · p.${citation.pages.join(', ')}`
      : '';

  return { href, label, pageText, isExternal: url };
}

/**
 * CitationList — a compact, non-intrusive sources strip. Collapsed by default
 * so it never blocks the answer; expands inline to reveal source pills.
 */
const CitationList = ({ citations, defaultOpen = false }) => {
  const [open, setOpen] = useState(defaultOpen);
  const hasCitations = Array.isArray(citations) && citations.length > 0;

  if (!hasCitations) {
    return null;
  }

  return (
    <div className={`citations ${open ? 'citations--open' : ''}`}>
      <button
        type="button"
        className="citations-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="citations-toggle-dot" aria-hidden="true" />
        <span className="citations-toggle-label">
          {citations.length} {citations.length === 1 ? 'source' : 'sources'}
        </span>
        <span className="citations-chevron" aria-hidden="true">
          {open ? '▾' : '▸'}
        </span>
      </button>

      {open && (
        <div className="citation-container" role="list">
          {citations.map((citation, index) => {
            const { href, label, pageText, isExternal } = resolveCitation(citation);
            return (
              <a
                key={`${citation.filename}-${index}`}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="citation-pill"
                title={isExternal ? citation.filename : `Open ${citation.filename}`}
                role="listitem"
              >
                <span className="citation-number">{index + 1}</span>
                <span className="citation-text">
                  {label}
                  {pageText}
                </span>
                {isExternal && (
                  <span className="citation-ext" aria-hidden="true">↗</span>
                )}
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default CitationList;
