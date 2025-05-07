import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt_logic import get_gpt_response  # Ensure this is correctly implemented

from bs4 import BeautifulSoup
import requests

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes and origins for now (you can restrict it later)
CORS(app, resources={r"/*": {"origins": "*"}})


# Health check endpoint
@app.route('/')
def index():
    return jsonify({"status": "Chatbot backend is running!"})


# GPT chat endpoint (general)
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_message = data['message']
        ai_reply = get_gpt_response(user_message, context_chunks=[])
        return jsonify({'response': ai_reply})

    except Exception as e:
        print(f"Error in /chat: {str(e)}")
        return jsonify({'error': str(e)}), 500


# GPT + website input endpoint
@app.route('/chat-with-url', methods=['POST'])
def chat_with_url():
    try:
        data = request.get_json()
        url = data.get('url')
        message = data.get('message')

        if not url or not message:
            return jsonify({'error': 'Both URL and message are required'}), 400

        # Fetch the web page
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx

        # Parse and clean HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text(separator=' ', strip=True)

        # Trim content to avoid GPT token limits
        context = page_text[:3000]

        # Send to GPT
        ai_reply = get_gpt_response(message, context_chunks=[context])
        return jsonify({'response': ai_reply})

    except requests.exceptions.RequestException as e:
        print(f"Request error in /chat-with-url: {e}")
        return jsonify({'error': 'Failed to fetch the provided URL'}), 400

    except Exception as e:
        print(f"Error in /chat-with-url: {e}")
        return jsonify({'error': str(e)}), 500


# Main


import tempfile
from werkzeug.utils import secure_filename
import numpy as np
import json
import faiss
import openai
from docx import Document as DocxDocument
import fitz  # PyMuPDF

# Initialize FAISS index (global)
embedding_dim = 1536  # For text-embedding-ada-002
index = faiss.IndexFlatL2(embedding_dim)
metadata_store = []

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[-1].lower()

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            text = extract_text_from_file(tmp.name, ext)

        # Chunk text
        chunks = chunk_text(text)

        # Embed & store
        embeddings = embed_chunks(chunks)
        store_in_faiss(embeddings, chunks)

        return jsonify({'status': f'File {filename} processed and stored successfully'})

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

def extract_text_from_file(filepath, ext):
    if ext == 'pdf':
        doc = fitz.open(filepath)
        return "\n".join([page.get_text() for page in doc])


    elif ext == 'docx':
        doc = DocxDocument(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type")
def chunk_text(text, max_tokens=500, overlap=50):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        token_count = len(sentence.split())
        if current_len + token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap:]  # Keep overlap
            current_len = len(current_chunk)
        current_chunk.append(sentence)
        current_len += token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        vector = response['data'][0]['embedding']
        embeddings.append(np.array(vector, dtype='float32'))
    return embeddings

def store_in_faiss(embeddings, chunks):
    global index, metadata_store
    index.add(np.vstack(embeddings))
    metadata_store.extend(chunks)
    # Optional: persist index to disk if needed
if __name__ == '__main__':
    app.run(debug=True)
