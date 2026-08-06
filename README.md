# AI-Powered Legal Document Simplification and Guidance (India)

API-first, decoupled legal assistant for Indian legal documents.

It accepts PDF uploads, extracts/segments clauses, classifies legal clause types, detects risks, retrieves relevant law from Chroma, and generates plain-English guidance with citations in a chatbot-style Streamlit UI.

## Disclaimer
This project is a legal-tech assistant prototype and not legal advice.

## What This System Does
1. Upload legal PDF.
2. Detect text vs scanned pages.
3. Extract text with `pdfplumber`, then OCR fallback using `pytesseract + PyMuPDF` for empty/scanned pages.
4. Segment into clauses by numbering + paragraph boundaries.
5. Chunk long clauses to token-safe windows.
6. Classify clauses using `law-ai/CustomInLawBERT` (heuristic fallback if weights are unavailable).
7. Run hybrid risk detection (rule + clause signal).
8. Retrieve top-k legal chunks from persistent Chroma using `BAAI/bge-small-en`.
9. Generate structured guidance via Ollama API (streaming supported).
10. Store per-user chat history in SQLite (Google sign-in).

## Architecture
### Backend (`FastAPI`)
- `POST /upload`: PDF processing + clause/risk analysis.
- `POST /ask`: non-stream RAG response.
- `POST /ask/stream`: token stream (SSE).
- `GET /auth/google/login`, `GET /auth/google/callback`, `GET /auth/me`.
- `GET /health`.

### Frontend (`Streamlit`)
- Google sign-in.
- Sidebar document upload + analysis trigger.
- Auto document explanation after analysis.
- Chat UI with citations and retrieved context.
- Multi-chat, new chat, delete chat.

### Data/State
- Vector DB: existing persistent Chroma in `vector_db/`.
- Local chat persistence: SQLite at `misc/chat_history.db` (or `CHAT_DB_PATH`).

## Folder Structure
```text
backend/
  api/
    routers/
frontend/
src/
  core/
  models/
  services/
    llm/
misc/
vector_db/
docker-compose.yml
```

## Key Technical Choices
- Embeddings: `BAAI/bge-small-en`
  - Query prefix: `search_query: `
  - Document prefix: `search_document: `
- Legal model: `law-ai/CustomInLawBERT`
  - If local path/repo weights are missing, it attempts download.
  - If still unavailable, it falls back to rule-based classification.
- LLM provider abstraction:
  - `ollama` (official `ollama` Python client)
  - `mock` fallback
- Retrieval:
  - default `TOP_K_RETRIEVAL=10`
  - reranking + dedupe + diversity-aware selection
  - state-aware citation filtering for India jurisdiction context

## Prerequisites
- Python `3.11` or `3.12`
- `uv` package manager
- Docker + Docker Compose (for container run)
- Tesseract OCR installed (local non-Docker runs)
  - Windows example: set `TESSERACT_CMD=C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`

## Local Run
Run these commands from the repository root.

1. Create and activate a virtual environment (PowerShell example):
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

2. Install dependencies:
```bash
uv sync # This installs all dependencies from pyproject.toml
```

3. Create `.env` from the example file and fill in secrets:
```powershell
Copy-Item .env.example .env
```

4. Start backend:
```bash
uv run python main.py
```
If you are already inside the activated virtual environment, this also works:
```bash
python main.py
```

5. Start frontend in a second terminal:
```bash
uv run streamlit run frontend/app.py
```
Or, with the activated venv:
```bash
python -m streamlit run frontend/app.py
```

6. Open:
- Frontend: `http://localhost:8501`
- Backend health: `http://localhost:8000/health`

## Docker Run
Build and run:
```bash
docker compose up --build
```

Open:
- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`

### Docker Notes
- Source bind mount is enabled: `.:/app` (code updates reflect without rebuilding image layers).
- `vector_db` is intentionally part of project mount; do not ignore it if you need the existing DB.
- Backend overrides model local paths to empty so models can resolve/download in container.
- Hugging Face cache is persisted in named volume `hf_cache`.
- Chat DB is persisted in named volume `chat_db` at `/app/misc`.

## Environment Variables
Core app:
- `APP_NAME`, `APP_ENV`, `LOG_LEVEL`

Vector/retrieval:
- `VECTOR_DB_PATH` (default `vector_db`)
- `CHROMA_COLLECTION`
- `TOP_K_RETRIEVAL` (default `10`)

Models:
- `EMBEDDING_MODEL_NAME` (default `BAAI/bge-small-en`)
- `EMBEDDING_MODEL_LOCAL_PATH` (optional local cache path)
- `LEGAL_MODEL_NAME` (default `law-ai/CustomInLawBERT`)
- `LEGAL_MODEL_LOCAL_PATH` (optional local cache path)
- `MAX_CLAUSE_TOKENS` (default `512`)

LLM:
- `LLM_PROVIDER` (`ollama` or `mock`)
- `OLLAMA_BASE_URL` (example: `https://ollama.com`)
- `OLLAMA_API_KEY`
- `OLLAMA_MODEL` (example: `gpt-oss:120b`)
- `OLLAMA_TEMPERATURE`
- `OLLAMA_TIMEOUT_SECONDS`

OCR:
- `TESSERACT_CMD` (required locally on Windows if not in PATH)
- `OCR_LANG`
- `OCR_DPI`

Auth/frontend:
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI` (backend callback URL)
- `FRONTEND_BASE_URL`
- `BACKEND_PUBLIC_URL`
- `BACKEND_BASE_URL` (frontend-to-backend URL)
- `AUTH_SECRET_KEY`
- `AUTH_TOKEN_TTL_SECONDS`

Chat DB:
- `CHAT_DB_PATH` (default `misc/chat_history.db`)

## API Reference
### `POST /upload`
`multipart/form-data` with `file` (PDF)

Response:
- `filename`
- `total_clauses`
- `summary`
- `clauses[]`: `clause_id`, `clause_text`, `clause_type`, `entities`, `confidence`
- `risks[]`: `clause_id`, `clause_text`, `clause_type`, `risk_level`, `triggers`

### `POST /ask`
JSON body:
```json
{
  "question": "What are termination risks?",
  "clause_text": "optional context",
  "top_k": 10
}
```

Response:
- `question`
- `answer`
- `retrieved_context[]`
- `citations[]` (`citation`, `section`, `reference_type`, `score`)

### `POST /ask/stream`
Same request body as `/ask`; returns `text/event-stream`.

Event types:
- `context`
- `token`
- `done`
- `error`

## OAuth Flow (Google)
1. Frontend opens backend login URL.
2. Backend redirects to Google.
3. Google callback hits backend `/auth/google/callback`.
4. Backend signs a short auth token and redirects to frontend with `auth_token` query param.
5. Frontend calls `/auth/me` with bearer token.
6. User identity is used to scope chats in local SQLite.

## Deployment Notes (Single Server)
Yes, both frontend and backend can run on one server with Docker.

Recommended production setup:
1. Put reverse proxy (Nginx/Caddy/Traefik) in front with HTTPS.
2. Set real public URLs:
   - `FRONTEND_BASE_URL=https://<domain>`
   - `BACKEND_PUBLIC_URL=https://<domain-or-api-subdomain>`
   - `GOOGLE_REDIRECT_URI=https://<backend-domain>/auth/google/callback`
3. Use persistent volumes for:
   - `vector_db`
   - `misc/chat_history.db`
   - Hugging Face cache
4. Store secrets in platform secret manager, not committed files.
5. Restrict CORS in production (current backend allows all origins).

## Performance Tips
- Keep `TOP_K_RETRIEVAL=10` unless latency constraints require lower values.
- Warm backend once after startup to load models.
- Use persistent HF cache to avoid repeated model download.
- OCR is expensive; native-text PDFs are much faster.
- For heavy traffic, add Redis for:
  - `/ask` response cache
  - retriever result cache
  - short-lived session/callback state

## Troubleshooting
### Port already in use
Error:
`[Errno 10048] ... bind on ('0.0.0.0', 8000)`

Fix:
- Stop existing process on port `8000`, or run backend on another port.

### `ModuleNotFoundError: No module named 'src'`
- Run from repo root.
- Use:
  - `uv run python main.py` for backend
  - `uv run streamlit run frontend/app.py` for frontend

### `Extracted 0 characters from PDF`
- The file may be scanned/image-based.
- Ensure Tesseract is installed and `TESSERACT_CMD` is valid.
- OCR fallback is implemented for scanned/empty pages.

### `Invalid OAuth state`
- Ensure Google OAuth callback points to backend `/auth/google/callback`.
- Verify `FRONTEND_BASE_URL`, `BACKEND_PUBLIC_URL`, and `GOOGLE_REDIRECT_URI` are consistent.
