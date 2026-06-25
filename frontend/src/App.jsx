import React, { useState, useEffect, useRef } from 'react';
import MarkdownMessage from './components/MarkdownMessage';
import CitationList from './components/CitationList';
import RetrievedChunks from './components/RetrievedChunks';
import PipelineStages from './components/PipelineStages';
import ThemeSwitcher from './components/ThemeSwitcher';
import { 
  queryConversationStream, 
  createConversation,
  getGlobalDocuments,
  uploadGlobalFile,
  deleteGlobalDocument
} from './services/api';

// Use the available hardware concurrency to upload/index files in parallel,
// capped to a sensible ceiling so we don't overwhelm the backend.
const UPLOAD_CONCURRENCY = Math.max(
  2,
  Math.min((typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4, 6)
);

/**
 * Run async `worker(item)` over `items` with a bounded number of concurrent
 * workers. Returns once every item has been processed.
 */
async function runWithConcurrency(items, limit, worker) {
  let cursor = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      await worker(items[index], index);
    }
  });
  await Promise.all(runners);
}

/**
 * Unified, professional, emoji-free CiteRAG application.
 * Splits the screen into a document management sidebar and an interactive chat pane.
 */
function App() {
  // --- Chat State ---
  const [inputText, setInputText] = useState('');
  const [submittedQuery, setSubmittedQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [citations, setCitations] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [isPending, setIsPending] = useState(false);
  const [chatError, setChatError] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  
  // --- Document Management State ---
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState({ text: '', type: '' });
  const [uploadProgress, setUploadProgress] = useState({ done: 0, total: 0 });
  const [fileStatuses, setFileStatuses] = useState([]); // [{ name, status }]
  const [deletingDocs, setDeletingDocs] = useState(new Set());
  const [isDragOver, setIsDragOver] = useState(false);

  const activeStreamRef = useRef(0);

  // --- Initialize Conversation and Fetch Documents ---
  useEffect(() => {
    // 1. Create conversation session
    createConversation()
      .then((data) => {
        setConversationId(data.conversation_id);
      })
      .catch((err) => {
        console.error('Failed to create conversation:', err);
      });

    // 2. Fetch already-ingested documents
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const data = await getGlobalDocuments();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  };

  // --- Document Upload Handlers ---
  const handleFileUploads = async (selectedFiles) => {
    if (!selectedFiles || selectedFiles.length === 0) return;

    const pdfFiles = selectedFiles.filter(file => file.name.toLowerCase().endsWith('.pdf'));
    const invalidFiles = selectedFiles.filter(file => !file.name.toLowerCase().endsWith('.pdf'));

    if (pdfFiles.length === 0) {
      showUploadMessage('Only PDF files are supported.', 'error');
      return;
    }

    setUploading(true);
    setUploadProgress({ done: 0, total: pdfFiles.length });
    setFileStatuses(pdfFiles.map(f => ({ name: f.name, status: 'queued' })));

    const successFiles = [];
    const failedFiles = [];
    let completed = 0;

    const updateStatus = (name, status) => {
      setFileStatuses(prev =>
        prev.map(fs => (fs.name === name ? { ...fs, status } : fs))
      );
    };

    // Upload and index files in parallel, bounded by the device's capacity.
    await runWithConcurrency(pdfFiles, UPLOAD_CONCURRENCY, async (file) => {
      updateStatus(file.name, 'uploading');
      try {
        const result = await uploadGlobalFile(file);
        if (result.success) {
          successFiles.push(file.name);
          updateStatus(file.name, 'done');
          fetchDocuments();
        } else {
          failedFiles.push(`${file.name} (failed to index)`);
          updateStatus(file.name, 'error');
        }
      } catch (err) {
        const errMsg = err.response?.data?.detail || err.message || 'unknown error';
        failedFiles.push(`${file.name} (${errMsg})`);
        updateStatus(file.name, 'error');
      } finally {
        completed += 1;
        setUploadProgress({ done: completed, total: pdfFiles.length });
      }
    });

    let messageText = '';
    let messageType = 'success';

    if (successFiles.length > 0) {
      messageText += `Indexed ${successFiles.length} file${successFiles.length !== 1 ? 's' : ''}. `;
    }
    if (failedFiles.length > 0) {
      messageText += `Failed: ${failedFiles.join(', ')}. `;
      messageType = successFiles.length > 0 ? 'info' : 'error';
    }
    if (invalidFiles.length > 0) {
      messageText += `Skipped non-PDF: ${invalidFiles.map(f => f.name).join(', ')}.`;
      if (messageType === 'success') messageType = 'info';
    }

    showUploadMessage(messageText.trim(), messageType);
    setUploading(false);
    setUploadProgress({ done: 0, total: 0 });
    // Clear the per-file status list shortly after completion.
    setTimeout(() => setFileStatuses([]), 4000);
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUploads(Array.from(e.target.files));
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUploads(Array.from(e.dataTransfer.files));
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDocumentDelete = async (filename) => {
    setDeletingDocs(prev => new Set([...prev, filename]));
    try {
      await deleteGlobalDocument(filename);
      fetchDocuments();
    } catch (err) {
      console.error('Failed to delete document:', err);
      alert(`Failed to delete document: ${err.message}`);
    } finally {
      setDeletingDocs(prev => {
        const next = new Set(prev);
        next.delete(filename);
        return next;
      });
    }
  };

  const showUploadMessage = (text, type) => {
    setUploadMessage({ text, type });
    if (type !== 'info') {
      setTimeout(() => {
        setUploadMessage({ text: '', type: '' });
      }, 5000);
    }
  };

  // --- Query Chat Handler ---
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
    setChatError(null);

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
      setChatError(err.message || 'The query could not be answered. Please try again.');
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
    <div className="container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div>
            <h1>CiteRAG</h1>
            <p>Unified Multi-Document QA Engine</p>
          </div>
          <div className="header-controls">
            <ThemeSwitcher />
          </div>
        </div>
      </header>

      {/* Main split-pane content */}
      <div className="split-pane">
        
        {/* Left Sidebar: Document Management */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <h2>Document Corpus</h2>
            
            {/* Upload Box */}
            <div 
              className={`drag-upload-box ${isDragOver ? 'dragover' : ''} ${uploading ? 'disabled' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => !uploading && document.getElementById('fileInput').click()}
            >
              <input 
                type="file" 
                id="fileInput" 
                style={{ display: 'none' }} 
                accept=".pdf"
                multiple
                onChange={handleFileChange}
                disabled={uploading}
              />
              <p>Drag and drop PDF files here, or click to choose files</p>
              <span className="upload-limit">
                Bulk upload supported · indexed {UPLOAD_CONCURRENCY} at a time
              </span>
            </div>

            {/* Bulk upload progress */}
            {uploading && uploadProgress.total > 0 && (
              <div className="upload-progress" role="status" aria-live="polite">
                <div className="upload-progress-head">
                  <span>Indexing documents</span>
                  <span>{uploadProgress.done} / {uploadProgress.total}</span>
                </div>
                <div className="upload-progress-track">
                  <div
                    className="upload-progress-bar"
                    style={{ width: `${(uploadProgress.done / uploadProgress.total) * 100}%` }}
                  />
                </div>
              </div>
            )}

            {/* Per-file status list */}
            {fileStatuses.length > 0 && (
              <ul className="upload-file-list">
                {fileStatuses.map((fs) => (
                  <li key={fs.name} className={`upload-file-item upload-file-item--${fs.status}`}>
                    <span className="upload-file-status-dot" aria-hidden="true" />
                    <span className="upload-file-name" title={fs.name}>{fs.name}</span>
                    <span className="upload-file-state">
                      {fs.status === 'queued' && 'Queued'}
                      {fs.status === 'uploading' && 'Indexing…'}
                      {fs.status === 'done' && 'Done'}
                      {fs.status === 'error' && 'Failed'}
                    </span>
                  </li>
                ))}
              </ul>
            )}

            {/* Upload Message */}
            {uploadMessage.text && (
              <div className={`status-msg ${uploadMessage.type}`}>
                {uploadMessage.text}
              </div>
            )}
          </div>

          {/* Ingested Documents List */}
          <div className="sidebar-section documents-list-section">
            <h3>Ingested Documents</h3>
            {documents.length === 0 ? (
              <p className="no-docs-msg">No documents in the corpus. Please upload a PDF above.</p>
            ) : (
              <div className="documents-scroll">
                {documents.map((doc, idx) => (
                  <div className="document-list-item" key={idx}>
                    <div className="document-info">
                      <span className="document-name" title={doc.filename}>{doc.filename}</span>
                    </div>
                    <button 
                      className="document-delete-btn"
                      onClick={() => handleDocumentDelete(doc.filename)}
                      disabled={deletingDocs.has(doc.filename)}
                    >
                      {deletingDocs.has(doc.filename) ? 'Deleting...' : 'Delete'}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>

        {/* Right Pane: Interactive Chatbot */}
        <main className="chat-container-main">
          {/* Pipeline stages */}
          <PipelineStages />

          {/* Chat Panel */}
          <section className="chat-panel" aria-label="Chat interface">
            
            {/* Chat History / Welcome Screen */}
            {!submittedQuery && !hasResult && !isPending && (
              <div className="welcome-chat-panel">
                <p>Welcome. Upload your PDF documents in the sidebar, then type a query below to retrieve answers grounded in the corpus.</p>
              </div>
            )}

            {/* Query submission and Answer block */}
            {(submittedQuery || hasResult || isPending) && (
              <div className="chat-thread">
                
                {/* User's Query */}
                {submittedQuery && (
                  <div className="chat-query-bubble">
                    <span className="bubble-label">Query</span>
                    <p>{submittedQuery}</p>
                  </div>
                )}

                {/* Answer Display */}
                {hasAnswer && (
                  <section className={`chat-answer ${isPending ? 'chat-answer--streaming' : ''}`} aria-label="Answer">
                    <div className="chat-answer-head">
                      <span className="chat-answer-avatar" aria-hidden="true">CR</span>
                      <h2 className="chat-answer-heading">Answer</h2>
                    </div>
                    <div className="chat-answer-body">
                      <MarkdownMessage content={answer} />
                    </div>

                    {/* Inline, non-intrusive sources directly under the answer */}
                    {!isPending && !chatError && citations.length > 0 && (
                      <div className="chat-answer-sources">
                        <CitationList citations={citations} />
                      </div>
                    )}
                  </section>
                )}

                {/* Pending indicator */}
                {isPending && (
                  <div className="chat-pending" role="status" aria-live="polite">
                    <span className="chat-spinner" aria-hidden="true" />
                    <span>Generating answer...</span>
                  </div>
                )}

                {/* Error Block */}
                {chatError && !isPending && (
                  <div className="chat-error" role="alert" aria-live="assertive">
                    <span>The query could not be answered. {chatError}</span>
                  </div>
                )}

                {/* Retrieved chunks */}
                {hasResult && !isPending && !chatError && (
                  <section className="chat-chunks" aria-label="Retrieved passages">
                    <RetrievedChunks chunks={chunks} />
                  </section>
                )}
              </div>
            )}

            {/* Input Form */}
            <form className="chat-form" onSubmit={handleSubmit} aria-label="Submit a question">
              <textarea
                id="query-input"
                className="chat-input"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your question here and press Enter..."
                rows={3}
                disabled={isPending}
                aria-label="Query input"
                aria-describedby={chatError ? 'chat-error' : undefined}
              />
              <button
                type="submit"
                className="chat-submit-btn"
                disabled={isPending || !inputText.trim()}
                aria-busy={isPending}
              >
                {isPending ? 'Generating...' : 'Ask'}
              </button>
            </form>
          </section>
        </main>

      </div>
    </div>
  );
}

export default App;
