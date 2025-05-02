# Minor change to trigger redeploy

from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt_logic import get_response  # This should be your custom logic

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000"])
  # Enable CORS for all routes and origins

# Health check endpoint
@app.route('/')
def index():
    return jsonify({"status": "Chatbot backend is running!"})

# Chat endpoint to receive user message and return AI response
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_message = data['message']
        ai_reply = get_response(user_message)

        return jsonify({'response': ai_reply})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
