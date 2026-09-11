"""Flask backend for a hybrid AI chatbot with URL and document context."""

import ipaddress
import os
import socket
import tempfile
from collections import deque
from urllib.parse import urljoin, urlparse

import faiss
import fitz
import numpy as np
import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI
from werkzeug.utils import secure_filename

from gpt_logic import get_gpt_response

app = Flask(__name__)
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:8000")
CORS(app, resources={r"/*": {"origins": [frontend_origin]}})

client = OpenAI()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536
index = faiss.IndexFlatL2(EMBEDDING_DIM)
metadata_store = []


def validate_public_url(url: str) -> None:
    """Reject non-HTTP(S), localhost and private-network URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP(S) URLs are allowed")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError("URL hostname could not be resolved") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private or local network URLs are not allowed")


def fetch_page_text(url: str) -> str:
    validate_public_url(url)
    response = requests.get(
        url,
        timeout=8,
        headers={"User-Agent": "HybridChatbotPortfolio/1.0"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def extract_text_from_file(filepath: str, ext: str) -> str:
    if ext == "pdf":
        with fitz.open(filepath) as doc:
            return "\n".join(page.get_text() for page in doc)
    if ext == "docx":
        doc = DocxDocument(filepath)
        return "\n".join(para.text for para in doc.paragraphs)
    if ext == "txt":
        with open(filepath, "r", encoding="utf-8") as handle:
            return handle.read()
    raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")


def chunk_text(text: str, chunk_words: int = 450, overlap_words: int = 50):
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    return [" ".join(words[i : i + chunk_words]) for i in range(0, len(words), step)]


def embed_texts(texts):
    if not texts:
        return []
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [np.array(item.embedding, dtype="float32") for item in response.data]


def store_in_faiss(embeddings, chunks):
    if not embeddings:
        return
    index.add(np.vstack(embeddings))
    metadata_store.extend(chunks)


@app.get("/")
def index_route():
    return jsonify({"status": "ok", "service": "AI Hybrid Chatbot Backend"})


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    try:
        return jsonify({"response": get_gpt_response(message, context_chunks=[])})
    except Exception as exc:
        app.logger.exception("Chat request failed")
        return jsonify({"error": "Chat request failed"}), 500


@app.post("/chat-with-url")
def chat_with_url():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    message = data.get("message", "").strip()
    if not url or not message:
        return jsonify({"error": "Both URL and message are required"}), 400
    try:
        context = fetch_page_text(url)[:6000]
        return jsonify({"response": get_gpt_response(message, [context])})
    except (ValueError, requests.RequestException) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("URL chat request failed")
        return jsonify({"error": "URL chat request failed"}), 500


@app.post("/upload")
def upload_file():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(uploaded.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"pdf", "docx", "txt"}:
        return jsonify({"error": "Unsupported file type. Use PDF, DOCX or TXT."}), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            temp_path = tmp.name
            uploaded.save(temp_path)
        text = extract_text_from_file(temp_path, ext)
        chunks = chunk_text(text)
        if not chunks:
            return jsonify({"error": "No readable text found in file"}), 400
        embeddings = embed_texts(chunks)
        store_in_faiss(embeddings, chunks)
        return jsonify({"status": "processed", "filename": filename, "chunks": len(chunks)})
    except Exception:
        app.logger.exception("Upload processing failed")
        return jsonify({"error": "File processing failed"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query-documents")
def query_documents():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not metadata_store:
        return jsonify({"error": "No documents have been indexed yet"}), 400

    try:
        query_vector = embed_texts([question])[0].reshape(1, -1)
        top_k = min(5, len(metadata_store))
        _, indices = index.search(query_vector, top_k)
        chunks = [metadata_store[i] for i in indices[0] if 0 <= i < len(metadata_store)]
        return jsonify({"response": get_gpt_response(question, chunks)})
    except Exception:
        app.logger.exception("Document query failed")
        return jsonify({"error": "Document query failed"}), 500


@app.post("/crawl-and-chat")
def crawl_and_chat():
    data = request.get_json(silent=True) or {}
    base_url = data.get("url", "").strip()
    message = data.get("message", "").strip()
    if not base_url or not message:
        return jsonify({"error": "Both URL and message are required"}), 400

    try:
        validate_public_url(base_url)
        base_host = urlparse(base_url).netloc
        visited = set()
        queue = deque([base_url])
        collected = []

        while queue and len(visited) < 5:
            current_url = queue.popleft()
            if current_url in visited:
                continue
            try:
                validate_public_url(current_url)
                response = requests.get(
                    current_url,
                    timeout=8,
                    headers={"User-Agent": "HybridChatbotPortfolio/1.0"},
                )
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                collected.append(soup.get_text(separator=" ", strip=True))
                visited.add(current_url)

                for link in soup.find_all("a", href=True):
                    candidate = urljoin(base_url, link["href"])
                    parsed = urlparse(candidate)
                    if parsed.scheme in {"http", "https"} and parsed.netloc == base_host:
                        if candidate not in visited:
                            queue.append(candidate)
            except (ValueError, requests.RequestException):
                continue

        context = "\n\n".join(collected)[:12000]
        if not context:
            return jsonify({"error": "No readable public pages could be fetched"}), 400
        return jsonify({"response": get_gpt_response(message, [context]), "pages": len(visited)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Crawl request failed")
        return jsonify({"error": "Crawl request failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
