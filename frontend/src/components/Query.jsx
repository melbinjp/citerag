import React, { useState, useContext, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { SessionContext } from '../contexts/SessionContext';
import { DocumentContext } from '../contexts/DocumentContext';
import { query, queryStream } from '../services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './Query.css';

const Query = () => {
  const { sessionId } = useContext(SessionContext);
  const { documents } = useContext(DocumentContext);
  const { t } = useTranslation();

  const getDocumentName = (docId) => {
    const doc = documents.find(d => d.doc_id === docId);
    if (!doc) return `Document ${docId.substring(0, 8)}...`;
    
    if (doc.name.startsWith('http')) {
      try {
        const url = new URL(doc.name);
        return url.hostname;
      } catch {
        return doc.name;
      }
    }
    return doc.name;
  };
  const [q, setQ] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [queryHistory, setQueryHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => {
    const savedHistory = localStorage.getItem('docqa-query-history');
    if (savedHistory) {
      setQueryHistory(JSON.parse(savedHistory));
    }
  }, []);

  const saveToHistory = (question, answer, sources) => {
    const historyItem = {
      id: Date.now(),
      question,
      answer,
      sources,
      timestamp: new Date().toLocaleString()
    };
    const newHistory = [historyItem, ...queryHistory.slice(0, 9)]; // Keep last 10
    setQueryHistory(newHistory);
    localStorage.setItem('docqa-query-history', JSON.stringify(newHistory));
  };

  const handleQueryChange = (e) => {
    setQ(e.target.value);
  };

  const handleQuery = async (question = q) => {
    if (!question) {
      setError(t('query.placeholder'));
      return;
    }
    setLoading(true);
    setError('');
    setResult({ answer: '', sources: [] });
    setQ(question); // Set the input to the question being asked

    try {
      let finalAnswer = '';
      let finalSources = [];
      const stream = queryStream(sessionId, question);
      
      for await (const chunk of stream) {
        if (chunk.type === 'sources') {
          finalSources = chunk.data;
          setResult(prev => ({ ...prev, sources: finalSources }));
        } else if (chunk.token) {
          finalAnswer += chunk.token;
          setResult(prev => ({ ...prev, answer: finalAnswer }));
        }
      }
      
      saveToHistory(question, finalAnswer, finalSources);
    } catch (err) {
      setError(t('query.error', { message: err.message }));
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const exampleQuestions = [
    "What is the main topic of the document?",
    "Summarize the key findings.",
    "What are the main conclusions?",
  ];

  return (
    <div className="query-container">
      <div className="query-input-area">
        <textarea
          value={q}
          onChange={handleQueryChange}
          placeholder={t('query.placeholder')}
          rows="4"
          className="query-textarea"
        />
        <button onClick={() => handleQuery()} disabled={loading || !q} className="query-button">
          {loading ? t('query.searching') : t('query.submit')}
        </button>
      </div>

      <div className="example-queries">
        <h4>Try these examples:</h4>
        {exampleQuestions.map((question, index) => (
          <span 
            key={index} 
            className="example-query" 
            onClick={() => handleQuery(question)}
          >
            {question}
          </span>
        ))}
      </div>

      {loading && <div className="loader">{t('query.searching')}</div>}
      {error && <div className="error-message">{error}</div>}

      {queryHistory.length > 0 && (
        <div className="history-section">
          <button 
            className="history-toggle"
            onClick={() => setShowHistory(!showHistory)}
          >
            📚 Query History ({queryHistory.length})
          </button>
          {showHistory && (
            <div className="history-list">
              {queryHistory.map((item) => (
                <div key={item.id} className="history-item" onClick={() => {
                  setQ(item.question);
                  setResult({ answer: item.answer, sources: item.sources });
                  setShowHistory(false);
                }}>
                  <div className="history-question">{item.question}</div>
                  <div className="history-time">{item.timestamp}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {result && (
        <div className="results-section">
          <div className="result-header">📝 Answer</div>
          <div className="answer-box pretext-paper">
            <div className="answer-text">
              <ReactMarkdown remarkPlugins={[remarkGfm]} children={result.answer} />
            </div>
          </div>
          
          {result.sources && result.sources.length > 0 && (
            <>
              <div className="result-header">📚 Sources</div>
              <div className="sources-box">
                {result.sources.map((source, index) => (
                  <div key={index} className="source-item">
                    <div className="source-text">"{source.text || source}"</div>
                    <div className="source-meta">
                      {source.score && (
                        <span>Confidence: {source.score > 1 ? source.score.toFixed(1) : (source.score * 100).toFixed(1)}%</span>
                      )}
                      {source.doc_id && (
                        <span className="source-doc">📄 {getDocumentName(source.doc_id)}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default Query;
