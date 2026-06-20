import React from 'react';
import './CitationList.css';

/**
 * CitationList renders a list of citation objects, each with a source PDF
 * filename and associated page numbers.
 *
 * Props:
 *   citations {Array} - array of { filename: string, pages: number[] }
 *
 * Requirements: 11.2, 11.3
 */
const CitationList = ({ citations }) => {
  const hasCitations = Array.isArray(citations) && citations.length > 0;

  if (!hasCitations) {
    return (
      <div className="citation-list citation-list--empty" role="status" aria-live="polite">
        <span className="citation-list__empty-icon" aria-hidden="true">📭</span>
        <span className="citation-list__empty-text">No sources available</span>
      </div>
    );
  }

  return (
    <section className="citation-list" aria-label="Citations">
      <ul className="citation-list__items" role="list">
        {citations.map((citation, index) => {
          const pageLabel =
            citation.pages && citation.pages.length > 0
              ? citation.pages.join(', ')
              : '—';

          return (
            <li key={`${citation.filename}-${index}`} className="citation-list__item">
              <span className="citation-list__icon" aria-hidden="true">📄</span>
              <div className="citation-list__body">
                <span className="citation-list__filename" title={citation.filename}>
                  {citation.filename}
                </span>
                <span className="citation-list__pages">
                  <span className="citation-list__pages-label">
                    {citation.pages && citation.pages.length === 1 ? 'Page:' : 'Pages:'}
                  </span>{' '}
                  {citation.pages && citation.pages.length > 0
                    ? citation.pages.map((page) => (
                        <span key={page} className="citation-list__page-badge" aria-label={`page ${page}`}>
                          {page}
                        </span>
                      ))
                    : <span className="citation-list__no-pages">—</span>
                  }
                </span>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
};

export default CitationList;
