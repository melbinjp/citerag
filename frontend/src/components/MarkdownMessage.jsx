import React, { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github.css';
import './MarkdownMessage.css';

/**
 * CodeBlock — a fenced code block with a language label and a copy button.
 * Inline code is rendered as a simple styled <code> element.
 */
function CodeBlock({ inline, className, children, ...props }) {
  const [copied, setCopied] = useState(false);

  const rawText = String(children).replace(/\n$/, '');
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';

  const handleCopy = useCallback(() => {
    const text = rawText;
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1600);
      });
    }
  }, [rawText]);

  if (inline) {
    return (
      <code className="md-inline-code" {...props}>
        {children}
      </code>
    );
  }

  return (
    <div className="md-code-block">
      <div className="md-code-header">
        <span className="md-code-lang">{language || 'text'}</span>
        <button
          type="button"
          className="md-code-copy"
          onClick={handleCopy}
          aria-label="Copy code to clipboard"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="md-code-pre">
        <code className={className} {...props}>
          {children}
        </code>
      </pre>
    </div>
  );
}

/**
 * MarkdownMessage — renders an assistant answer with GitHub-flavoured markdown,
 * syntax-highlighted code, styled tables, images, and safe external links.
 * Designed to render any kind of rich content like a modern chat assistant.
 */
const MarkdownMessage = ({ content }) => {
  return (
    <div className="md-message">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          code: CodeBlock,
          a: ({ children, href, ...props }) => (
            <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
              {children}
            </a>
          ),
          table: ({ children, ...props }) => (
            <div className="md-table-wrap">
              <table {...props}>{children}</table>
            </div>
          ),
          img: ({ alt, ...props }) => (
            <img className="md-image" loading="lazy" alt={alt || ''} {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownMessage;
