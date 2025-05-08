import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from gpt_logic import get_gpt_response

from bs4 import BeautifulSoup
import requests

app = Flask(__name__)
CORS(app, supports_credentials=True)  # ✅ Strong CORS

@app.route('/')
def index():
    return jsonify({"status": "Chatbot backend is running!"})

@app.route('/chat', methods=['POST', 'OPTIONS'])
@cross_origin()
def chat():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200
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

@app.route('/chat-with-url', methods=['POST', 'OPTIONS'])
@cross_origin()
def chat_with_url():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200
    try:
        data = request.get_json()
        url = data.get('url')
        message = data.get('message')
        if not url or not message:
            return jsonify({'error': 'Both URL and message are required'}), 400
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text(separator=' ', strip=True)
        context = page_text[:3000]
        ai_reply = get_gpt_response(message, context_chunks=[context])
        return jsonify({'response': ai_reply})
    except requests.exceptions.RequestException as e:
        print(f"Request error in /chat-with-url: {e}")
        return jsonify({'error': 'Failed to fetch the provided URL'}), 400
    except Exception as e:
        print(f"Error in /chat-with-url: {e}")
        return jsonify({'error': str(e)}), 500

import tempfile
from werkzeug.utils import secure_filename
import numpy as np
import faiss
import openai
from docx import Document as DocxDocument
import fitz
from urllib.parse import urljoin, urlparse
from collections import deque

embedding_dim = 1536
index = faiss.IndexFlatL2(embedding_dim)
metadata_store = []

@app.route('/upload', methods=['POST', 'OPTIONS'])
@cross_origin()
def upload_file():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            text = extract_text_from_file(tmp.name, ext)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        store_in_faiss(embeddings, chunks)
        print(f"✅ Uploaded and stored {len(embeddings)} chunks in FAISS.")  # ✅ Debug print
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
            current_chunk = current_chunk[-overlap:]
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

@app.route('/query-documents', methods=['POST', 'OPTIONS'])
@cross_origin()
def query_documents():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200
    try:
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        response = openai.Embedding.create(
            input=question,
            model="text-embedding-ada-002"
        )
        query_vector = np.array(response['data'][0]['embedding'], dtype='float32').reshape(1, -1)
        top_k = 5
        D, I = index.search(query_vector, top_k)
        matched_chunks = [metadata_store[i] for i in I[0] if i < len(metadata_store)]
        print("🔍 Matched Chunks:\n", matched_chunks)  # ✅ Debug print
        ai_reply = get_gpt_response(question, context_chunks=matched_chunks)
        return jsonify({'response': ai_reply})
    except Exception as e:
        print(f"Error in /query-documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/crawl-and-chat', methods=['POST', 'OPTIONS'])
@cross_origin()
def crawl_and_chat():
    print(f"✅ Received request at /crawl-and-chat with method: {request.method}")
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200
    try:
        data = request.get_json()
        base_url = data.get('url')
        message = data.get('message')
        if not base_url or not message:
            return jsonify({'error': 'Both URL and message are required'}), 400
        max_pages = 5  # You can increase if needed
        visited = set()
        queue = deque([base_url])
        all_text = ""
        while queue and len(visited) < max_pages:
            current_url = queue.popleft()
            if current_url in visited:
                continue
            try:
                response = requests.get(current_url, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                page_text = soup.get_text(separator=' ', strip=True)
                all_text += page_text + "\n\n"
                visited.add(current_url)
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    parsed_href = urljoin(base_url, href)
                    if (urlparse(parsed_href).netloc == urlparse(base_url).netloc and
                        parsed_href not in visited and
                        parsed_href.startswith('http')):
                        queue.append(parsed_href)
            except Exception as e:
                error_msg = f"Failed to fetch {current_url}: {e}"
                print(error_msg)
                continue
        context = all_text[:8000]
        ai_reply = get_gpt_response(message, context_chunks=[context])
        return jsonify({'response': ai_reply})
    except Exception as e:
        print(f"Error in /crawl-and-chat: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
