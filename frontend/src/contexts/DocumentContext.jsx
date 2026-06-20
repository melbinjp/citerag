import React, { useState, createContext, useEffect } from 'react';

export const DocumentContext = createContext();

export const DocumentProvider = ({ children }) => {
  const [documents, setDocuments] = useState([]);

  useEffect(() => {
    const savedDocs = localStorage.getItem('docqa-documents');
    if (savedDocs) {
      setDocuments(JSON.parse(savedDocs));
    }
  }, []);

  const saveToStorage = (docs) => {
    localStorage.setItem('docqa-documents', JSON.stringify(docs));
  };

  const addDocument = (doc) => {
    const newDocs = [...documents, doc];
    setDocuments(newDocs);
    saveToStorage(newDocs);
  };

  const removeDocument = (docId) => {
    const newDocs = documents.filter((doc) => doc.doc_id !== docId);
    setDocuments(newDocs);
    saveToStorage(newDocs);
  };

  return (
    <DocumentContext.Provider value={{ documents, addDocument, removeDocument }}>
      {children}
    </DocumentContext.Provider>
  );
};
