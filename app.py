from fastapi import FastAPI
from pydantic import BaseModel
import openai
from vector_store import get_relevant_chunks
from gpt_logic import get_gpt_response
import os

# ✅ CORS middleware import
from fastapi.middleware.cors import CORSMiddleware

# 🔧 Create FastAPI app
app = FastAPI()

# ✅ CORS setup — this must come immediately after app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 Load OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📦 Define request body format
class ChatRequest(BaseModel):
    message: str

# 🤖 POST endpoint for chatbot
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message
    context_chunks = get_relevant_chunks(user_message)
    answer = get_gpt_response(user_message, context_chunks)
    return {"answer": answer}

# 🧪 GET test route (for debugging and CORS testing)
@app.get("/")
async def root():
    return {"message": "Hello from chatbot backend!"}
