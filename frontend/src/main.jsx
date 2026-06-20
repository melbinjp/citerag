import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter } from 'react-router-dom';
import App from './App.jsx'
import './index.css'
import './i18n';
import { SessionProvider } from './contexts/SessionContext.jsx';
import { DocumentProvider } from './contexts/DocumentContext.jsx';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <SessionProvider>
      <DocumentProvider>
        <HashRouter>
          <App />
        </HashRouter>
      </DocumentProvider>
    </SessionProvider>
  </React.StrictMode>,
)
