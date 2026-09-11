# AI Hybrid Chatbot Backend

A Flask-based backend for a hybrid AI assistant that can answer from general model knowledge, uploaded documents, a single public web page, or a small same-domain website crawl.

## What it demonstrates

- REST API design with Flask
- OpenAI Responses API integration
- Retrieval-augmented generation using FAISS
- PDF, DOCX and TXT ingestion
- Embedding-based document search
- Public URL extraction and bounded website crawling
- Environment-based configuration
- Basic URL safety controls to reduce server-side request forgery risk
- Deployment configuration for Render/Gunicorn

## Architecture

```text
Frontend
   |
   v
Flask API
   |---------------------> OpenAI model
   |
   +--> Uploaded document --> chunking --> embeddings --> FAISS
   |                                              |
   |                                              v
   +-------------------------------------- retrieved context
   |
   +--> Public URL / bounded site crawl --------> context
```

## API endpoints

- `GET /` - health/status endpoint
- `POST /chat` - standard AI chat
- `POST /chat-with-url` - answer using text from one public URL
- `POST /upload` - ingest a PDF, DOCX or TXT file into the in-memory vector index
- `POST /query-documents` - retrieve relevant document chunks and answer a question
- `POST /crawl-and-chat` - crawl up to five pages on one public domain and answer using that context

## Technology

Python, Flask, OpenAI API, FAISS, NumPy, Beautiful Soup, PyMuPDF, python-docx, Requests and Gunicorn.

## Run locally

```bash
git clone https://github.com/tsyed1409/i-hybrid-chatbot-backend.git
cd i-hybrid-chatbot-backend
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` or set environment variables based on `.env.example`. At minimum, set `OPENAI_API_KEY`.

Then run:

```bash
python app.py
```

The API will be available at `http://localhost:5000` by default.

## Example request

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain retrieval-augmented generation simply."}'
```

## Security and design notes

API credentials are read from environment variables and are not committed to the repository. URL-based endpoints only accept public HTTP(S) hosts and reject local/private network destinations. Uploaded temporary files are deleted after processing.

The document index is intentionally in-memory for this portfolio implementation. A production version would normally use persistent storage, authentication, rate limiting, structured observability, file-size limits, stronger content validation, and a managed vector database where appropriate.

## Limitations

- Uploaded document vectors are lost when the process restarts.
- The website crawler is intentionally limited to five same-domain pages.
- Retrieval quality depends on source content and embedding similarity.
- Generated answers can still be incorrect and should be evaluated for the intended use case.

## Related project

This repository provides the backend service for the separate `ai-chatbot-frontend` repository.

## Author

**Tariq Syed** - AI, product and digital transformation practitioner.
