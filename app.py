# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import openai
from vector_store import get_relevant_chunks
from gpt_logic import get_gpt_response
import os

# ✅ ADD CORS IMPORT
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Request model
class ChatRequest(BaseModel):
    message: str

# Chat endpoint
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message
    context_chunks = get_relevant_chunks(user_message)
    answer = get_gpt_response(user_message, context_chunks)
    return {"answer": answer}
