// Local dev uses backend on localhost:3001 (not using Render)
const API_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:8000/api/v1/analysis/submit'
    : 'http://localhost:8000/api/v1/analysis/submit';
