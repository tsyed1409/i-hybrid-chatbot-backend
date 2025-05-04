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
if __name__ == '__main__':
    app.run(debug=True)
