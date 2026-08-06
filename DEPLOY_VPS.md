# VPS Deployment Guide

This project is designed to run well on a single Linux VPS with Docker Compose.

## Recommended VPS

- Ubuntu 24.04 LTS
- 2 vCPU minimum
- 4 GB RAM minimum
- 40 GB SSD minimum

Use more RAM if you expect larger traffic or frequent cold starts.

## What You Need Before Starting

- A VPS provider and a new Ubuntu server
- The server public IP
- SSH access to the server
- Your domain name, if you want a clean URL
- Google Cloud Console access to update OAuth redirect URIs
- A fresh Ollama API key
- A local copy of `vector_db/vector_db`

## 1. Connect To The Server

From your local machine:

```bash
ssh root@YOUR_SERVER_IP
```

If your provider gives a non-root user, use that user instead.

## 2. Install Docker And Compose

Follow Docker's official Ubuntu installation docs:
- https://docs.docker.com/engine/install/ubuntu/
- https://docs.docker.com/compose/install/linux/

After install, verify:

```bash
docker --version
docker compose version
```

## 3. Clone The Repo

```bash
git clone https://github.com/Atharva7890ghosalkar/AI-Powered-Legal-Document-Simplification-and-Guidance-System.git
cd AI-Powered-Legal-Document-Simplification-and-Guidance-System
```

## 4. Create Production .env

Use `.env.example` as the starting point and create `.env`.

Set these values for production:

```env
APP_ENV=prod
LOG_LEVEL=INFO
AUTH_SECRET_KEY=generate_a_long_random_secret

FRONTEND_BASE_URL=https://YOUR_DOMAIN
BACKEND_PUBLIC_URL=https://YOUR_API_DOMAIN
BACKEND_BASE_URL=http://backend:8000

VECTOR_DB_PATH=vector_db/vector_db
CHROMA_COLLECTION=indian_legal_statutes

GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=https://YOUR_API_DOMAIN/auth/google/callback

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_new_ollama_api_key
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_TEMPERATURE=0.2
OLLAMA_TIMEOUT_SECONDS=120

OCR_LANG=eng
OCR_DPI=300
```

Note:
- `TESSERACT_CMD` is already overridden to `/usr/bin/tesseract` for the backend container in `docker-compose.yml`.
- `BACKEND_BASE_URL` should stay `http://backend:8000` for the frontend container.

## 5. Upload The Vector DB

Your GitHub repo does not include the vector DB. Copy it manually to the server.

From your local machine:

```bash
scp -r vector_db root@YOUR_SERVER_IP:/root/AI-Powered-Legal-Document-Simplification-and-Guidance-System/
```

After upload, the server should contain:

```text
vector_db/
  vector_db/
    chroma.sqlite3
    <collection data directory>
```

## 6. Start The App

```bash
docker compose up -d --build
```

Check status:

```bash
docker compose ps
docker compose logs --tail=100 backend
docker compose logs --tail=100 frontend
```

## 7. Verify It Works

Backend health:

```bash
curl http://localhost:8000/health
```

Frontend:

Open:

```text
http://YOUR_SERVER_IP:8501
```

## 8. Add Domain And HTTPS

Recommended:
- `app.yourdomain.com` -> frontend
- `api.yourdomain.com` -> backend

Use a reverse proxy such as Nginx or Caddy in front of the containers.

After domain setup, update:

- `FRONTEND_BASE_URL`
- `BACKEND_PUBLIC_URL`
- `GOOGLE_REDIRECT_URI`

## 9. Update Google OAuth

In Google Cloud Console, add the production callback URI:

```text
https://YOUR_API_DOMAIN/auth/google/callback
```

Also ensure your app/frontend domain is allowed.

## 10. Rotate Secrets

Before public deployment:

- revoke the old Ollama API key
- create a new Ollama API key
- use the new key in production `.env`

## Quick Troubleshooting

### Frontend loads but chat fails

Check:

```bash
docker compose logs --tail=200 backend
```

### Vector DB errors

Check that this exists:

```text
vector_db/vector_db/chroma.sqlite3
```

And confirm `.env` contains:

```env
VECTOR_DB_PATH=vector_db/vector_db
CHROMA_COLLECTION=indian_legal_statutes
```

### Google login fails

Check that production URLs match exactly in:

- `.env`
- Google Cloud Console

### OCR fails on scanned PDFs

Check backend container:

```bash
docker compose exec backend tesseract --version
```
