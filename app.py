# Minor change to trigger redeploy


import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt_logic import get_gpt_response  # ✅ Correct

app = Flask(__name__)

# ✅ Enable CORS for your local frontend
frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://localhost:8000")
CORS(app, origins=["null", "http://localhost:8000"], supports_credentials=True)


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
        ai_reply = get_gpt_response(user_message, context_chunks=[])  # ✅ new

        return jsonify({'response': ai_reply})

       except Exception as e:
        print(f"Error in /chat: {str(e)}")
        return jsonify({'error': str(e)}), 500



