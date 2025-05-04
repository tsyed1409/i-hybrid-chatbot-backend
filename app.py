# Minor change to trigger redeploy

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt_logic import get_gpt_response  # ✅ Correct

app = Flask(__name__)

# ✅ Enable CORS for local and file:// testing
frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://localhost:8000")
CORS(app, origins=[
    "null",
    "http://localhost:8000",
    "https://ai-chatbot-frontend-s5lk.onrender.com",  # ✅ your live frontend domain
], supports_credentials=True)


# ✅ Health check endpoint
@app.route('/')
def index():
    return jsonify({"status": "Chatbot backend is running!"})

# ✅ Chat endpoint
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

from bs4 import BeautifulSoup
import requests

@app.route('/chat-with-url', methods=['POST'])
def chat_with_url():
    try:
        data = request.get_json()
        url = data.get('url')
        message = data.get('message')

        if not url or not message:
            return jsonify({'error': 'Both URL and message are required'}), 400

        # Fetch the web page
        page = requests.get(url, timeout=5)
        soup = BeautifulSoup(page.text, 'html.parser')

        # Clean and extract the text
        page_text = soup.get_text(separator=' ', strip=True)

        # Trim to avoid going over GPT token limits
        context = page_text[:3000]

        # Use the same GPT function with webpage context
        ai_reply = get_gpt_response(message, context_chunks=[context])

        return jsonify({'response': ai_reply})

    except Exception as e:
        print(f"Error in /chat-with-url: {e}")
        return jsonify({'error': str(e)}), 500

