import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import ErrorBoundary from './ErrorBoundary.jsx'

const rootElement = document.getElementById('root');

if (!rootElement) {
  console.error('Root element not found!');
  document.body.innerHTML = '<div style="padding: 20px;">Error: Root element not found. Please check your HTML.</div>';
} else {
  try {
    const root = createRoot(rootElement);
    root.render(
      <StrictMode>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </StrictMode>
    );
  } catch (error) {
    console.error('Fatal error rendering app:', error);
    rootElement.innerHTML = `
      <div style="padding: 40px; font-family: Arial, sans-serif; text-align: center;">
        <h1 style="color: #d32f2f;">Fatal Error</h1>
        <p>${error.message || 'Unknown error'}</p>
        <button onclick="window.location.reload()" style="padding: 12px 24px; background: #ff9900; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 20px;">
          Reload Page
        </button>
      </div>
    `;
  }
}
