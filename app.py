# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import openai
from vector_store import get_relevant_chunks
from gpt_logic import get_gpt_response
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message
    context_chunks = get_relevant_chunks(user_message)
    answer = get_gpt_response(user_message, context_chunks)
    return {"answer": answer}
