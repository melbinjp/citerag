import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    debug: true,
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false, // not needed for react as it escapes by default
    },
    resources: {
      en: {
        translation: {
          nav: {
            upload: 'Upload',
            query: 'Query',
            documents: 'Documents',
          },
          upload: {
            title: 'Upload Documents',
            selectFileOrUrl: 'Please select a file or enter a URL.',
            dragAndDrop: 'Drag and drop a file here',
            orClick: 'or click to select a file',
            or: 'or',
            enterUrl: 'Enter a URL',
            processing: 'Processing...',
            submit: 'Submit',
            successFile: 'File uploaded successfully! Doc ID: {{docId}}',
            successUrl: 'URL ingested successfully! Doc ID: {{docId}}',
            error: 'Error: {{message}}'
          },
          query: {
            title: 'Ask a Question',
            placeholder: 'Enter your question',
            submit: 'Submit Query',
            searching: 'Searching...',
            error: 'Error performing query: {{message}}',
            answer: 'Answer',
            sources: 'Sources',
          },
          documents: {
            title: 'My Documents',
            noDocuments: 'No documents uploaded yet.',
            delete: 'Delete',
            error: 'Error deleting document: {{message}}',
          },
          session: {
            loading: 'Loading session...',
            id: 'Session ID: {{sessionId}}',
          }
        }
      },
      es: {
        translation: {
          nav: {
            upload: 'Subir',
            query: 'Preguntar',
            documents: 'Documentos',
          },
          upload: {
            title: 'Subir Documentos',
            selectFileOrUrl: 'Por favor seleccione un archivo o ingrese una URL.',
            dragAndDrop: 'Arrastre y suelte un archivo aquí',
            orClick: 'o haga clic para seleccionar un archivo',
            or: 'o',
            enterUrl: 'Ingrese una URL',
            processing: 'Procesando...',
            submit: 'Enviar',
            successFile: '¡Archivo subido con éxito! ID del documento: {{docId}}',
            successUrl: '¡URL ingerida con éxito! ID del documento: {{docId}}',
            error: 'Error: {{message}}'
          },
          query: {
            title: 'Hacer una Pregunta',
            placeholder: 'Escriba su pregunta',
            submit: 'Enviar Pregunta',
            searching: 'Buscando...',
            error: 'Error al realizar la consulta: {{message}}',
            answer: 'Respuesta',
            sources: 'Fuentes',
          },
          documents: {
            title: 'Mis Documentos',
            noDocuments: 'Aún no se han subido documentos.',
            delete: 'Eliminar',
            error: 'Error al eliminar el documento: {{message}}',
          },
          session: {
            loading: 'Cargando sesión...',
            id: 'ID de sesión: {{sessionId}}',
          }
        }
      }
    }
  });

export default i18n;