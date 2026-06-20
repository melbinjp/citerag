import React, { useContext, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { SessionContext } from '../contexts/SessionContext';
import { DocumentContext } from '../contexts/DocumentContext';
import { deleteDocument } from '../services/api';
import './Documents.css';

const Documents = () => {
  const { sessionId } = useContext(SessionContext);
  const { documents, removeDocument } = useContext(DocumentContext);
  const { t } = useTranslation();
  const [error, setError] = useState('');
  const [deletingDocs, setDeletingDocs] = useState(new Set());

  const handleDelete = async (docId) => {
    setDeletingDocs(prev => new Set([...prev, docId]));
    setError('');
    
    try {
      await deleteDocument(sessionId, docId);
      removeDocument(docId);
    } catch (err) {
      setError(`Failed to delete document: ${err.message}`);
    } finally {
      setDeletingDocs(prev => {
        const newSet = new Set(prev);
        newSet.delete(docId);
        return newSet;
      });
    }
  };

  const getDisplayName = (doc) => {
    if (doc.name) {
      // For URLs, show domain name
      if (doc.name.startsWith('http')) {
        try {
          const url = new URL(doc.name);
          return url.hostname + url.pathname;
        } catch {
          return doc.name;
        }
      }
      return doc.name;
    }
    return `Document ${doc.doc_id.substring(0, 8)}...`;
  };

  const getDocumentType = (doc) => {
    if (doc.name) {
      if (doc.name.startsWith('http')) return 'URL';
      if (doc.name.endsWith('.pdf')) return 'PDF';
      if (doc.name.endsWith('.docx')) return 'DOCX';
      if (doc.name.endsWith('.txt')) return 'TXT';
    }
    return 'Document';
  };

  return (
    <div className="documents-section">
      <h3>📚 My Documents</h3>
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      {documents.length === 0 ? (
        <p style={{ color: '#666', textAlign: 'center', padding: '40px' }}>
          No documents uploaded yet. Go to the Upload tab to add documents.
        </p>
      ) : (
        <div className="documents-list">
          {documents.map((doc) => (
            <div className="doc-item" key={doc.doc_id}>
              <div className="doc-details">
                <div className="doc-name">{getDisplayName(doc)}</div>
                <div className="doc-info">
                  {getDocumentType(doc)} • {doc.num_chunks || 0} chunks
                </div>
              </div>
              <button 
                className="delete-btn"
                onClick={() => handleDelete(doc.doc_id)}
                disabled={deletingDocs.has(doc.doc_id)}
                title="Delete document"
              >
                {deletingDocs.has(doc.doc_id) ? '⏳' : '🗑️'}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Documents;