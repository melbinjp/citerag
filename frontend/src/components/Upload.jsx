import React, { useState, useContext, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { SessionContext } from '../contexts/SessionContext';
import { DocumentContext } from '../contexts/DocumentContext';
import { ingestFile, ingestUrl } from '../services/api';
import './Upload.css';

const Upload = () => {
  const { sessionId } = useContext(SessionContext);
  const { addDocument } = useContext(DocumentContext);
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [urlInput, setUrlInput] = useState('');
  const [urls, setUrls] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [processResults, setProcessResults] = useState([]);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(prev => [...prev, ...selectedFiles]);
  };

  const handleUrlInputChange = (e) => {
    const value = e.target.value;
    setUrlInput(value);
    
    // Parse comma-separated URLs
    const urlList = value.split(',')
      .map(url => url.trim())
      .filter(url => url.length > 0);
    setUrls(urlList);
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const removeUrl = (index) => {
    const newUrls = urls.filter((_, i) => i !== index);
    setUrls(newUrls);
    setUrlInput(newUrls.join(', '));
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      setFiles(prev => [...prev, ...droppedFiles]);
      e.dataTransfer.clearData();
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const showMessage = (text, type) => {
    setMessage(text);
    setMessageType(type);
    setTimeout(() => {
      setMessage('');
      setMessageType('');
    }, 5000);
  };

  const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  const processWithRetry = async (processFunc, item, itemName, maxRetries = 2) => {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Processing ${itemName} (attempt ${attempt}), Session: ${sessionId}`);
        const response = await processFunc();
        return { success: true, response };
      } catch (error) {
        console.error(`Error processing ${itemName} (attempt ${attempt}):`, error);
        
        if (attempt === maxRetries) {
          return { 
            success: false, 
            error: error.response?.data?.detail || error.message 
          };
        }
        
        // Wait before retry (exponential backoff)
        await delay(1000 * attempt);
      }
    }
  };

  const handleSubmit = async () => {
    if (files.length === 0 && urls.length === 0) {
      showMessage('Please select files or enter URLs', 'error');
      return;
    }

    setLoading(true);
    setProcessResults([]);
    let successCount = 0;
    let errorCount = 0;
    const errors = [];

    try {
      // Process files with delay
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        const maxSize = 5 * 1024 * 1024; // 5MB
        if (file.size > maxSize) {
          errors.push(`${file.name}: File too large (max 5MB)`);
          errorCount++;
          continue;
        }
        
        const result = await processWithRetry(
          () => ingestFile(sessionId, file),
          file,
          file.name
        );
        
        if (result.success) {
          addDocument({ ...result.response, name: file.name });
          successCount++;
          setProcessResults(prev => [...prev, {
            name: file.name,
            type: 'file',
            status: 'success',
            docId: result.response.doc_id
          }]);
        } else {
          errors.push(`${file.name}: ${result.error}`);
          errorCount++;
          setProcessResults(prev => [...prev, {
            name: file.name,
            type: 'file',
            status: 'error',
            error: result.error
          }]);
        }
        
        // Add delay between requests to avoid rate limiting
        if (i < files.length - 1 || urls.length > 0) {
          await delay(500);
        }
      }

      // Process URLs with delay
      for (let i = 0; i < urls.length; i++) {
        const url = urls[i];
        
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
          errors.push(`${url}: Invalid URL format`);
          errorCount++;
          continue;
        }
        
        const result = await processWithRetry(
          () => ingestUrl(sessionId, url),
          url,
          url
        );
        
        if (result.success) {
          addDocument({ ...result.response, name: url });
          successCount++;
          setProcessResults(prev => [...prev, {
            name: url,
            type: 'url',
            status: 'success',
            docId: result.response.doc_id
          }]);
        } else {
          errors.push(`${url}: ${result.error}`);
          errorCount++;
          setProcessResults(prev => [...prev, {
            name: url,
            type: 'url',
            status: 'error',
            error: result.error
          }]);
        }
        
        // Add delay between requests
        if (i < urls.length - 1) {
          await delay(500);
        }
      }

      // Show summary
      if (successCount > 0 && errorCount === 0) {
        showMessage(`✅ All ${successCount} items processed successfully!`, 'success');
      } else if (successCount > 0 && errorCount > 0) {
        showMessage(`⚠️ ${successCount} succeeded, ${errorCount} failed`, 'error');
      } else if (errorCount > 0) {
        showMessage(`❌ All ${errorCount} items failed`, 'error');
      }
      
      // Clear form only if all succeeded
      if (errorCount === 0) {
        setFiles([]);
        setUrls([]);
        setUrlInput('');
      }
      
    } catch (error) {
      console.error('Unexpected error:', error);
      showMessage(`Unexpected error: ${error.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-tab">
      <div 
        className={`upload-section ${isDragOver ? 'dragover' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div style={{ fontSize: '48px', marginBottom: '20px' }}>📁</div>
        <h3>Drop your files here or click to upload</h3>
        <p style={{ color: '#666', margin: '10px 0' }}>Supports PDF, DOCX, TXT, XLSX, CSV, PPTX, HTML, MD (max 5MB)</p>
        
        <input 
          type="file" 
          id="fileInput" 
          style={{ display: 'none' }} 
          accept=".pdf,.docx,.txt,.xlsx,.csv,.pptx,.html,.htm,.md"
          multiple
          onChange={handleFileChange}
        />
        <button 
          className="upload-btn" 
          onClick={() => document.getElementById('fileInput').click()}
        >
          Choose Files
        </button>
      </div>

      {files.length > 0 && (
        <div className="file-list">
          <h4>Selected Files:</h4>
          {files.map((file, index) => (
            <div key={index} className="file-item">
              <span>{file.name}</span>
              <button onClick={() => removeFile(index)} className="remove-btn">×</button>
            </div>
          ))}
        </div>
      )}

      <div style={{ textAlign: 'center', margin: '20px 0', color: '#666' }}>
        — OR —
      </div>

      <div className="url-input-group">
        <input 
          type="text" 
          value={urlInput}
          onChange={handleUrlInputChange}
          className="url-input" 
          placeholder="Enter URLs separated by commas..."
        />
      </div>

      {urls.length > 0 && (
        <div className="url-list">
          <h4>URLs to process:</h4>
          {urls.map((url, index) => (
            <div key={index} className="url-item">
              <span>{url}</span>
              <button onClick={() => removeUrl(index)} className="remove-btn">×</button>
            </div>
          ))}
        </div>
      )}

      <button 
        className="submit-btn" 
        onClick={handleSubmit} 
        disabled={loading || (files.length === 0 && urls.length === 0)}
      >
        {loading ? 'Processing...' : 'Submit All'}
      </button>

      {message && (
        <div className={`status-message show ${messageType}`}>
          {message}
        </div>
      )}

      {processResults.length > 0 && (
        <div className="process-results">
          <h4>Processing Results:</h4>
          {processResults.map((result, index) => (
            <div key={index} className={`result-item ${result.status}`}>
              <div className="result-info">
                <span className="result-icon">
                  {result.status === 'success' ? '✅' : '❌'}
                </span>
                <span className="result-name">{result.name}</span>
                <span className="result-type">({result.type})</span>
              </div>
              {result.status === 'success' && (
                <span className="result-doc-id">ID: {result.docId}</span>
              )}
              {result.status === 'error' && (
                <span className="result-error">{result.error}</span>
              )}
            </div>
          ))}
          
          {processResults.some(r => r.status === 'success') && (
            <div className="query-prompt">
              <p>✨ Ready to ask questions about your documents?</p>
              <button 
                className="go-to-query-btn" 
                onClick={() => navigate('/query')}
              >
                💬 Go to Query Section
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Upload;