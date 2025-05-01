from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
from vector_store import get_relevant_chunks
from gpt_logic import get_gpt_response

app = FastAPI()

# ✅ CORS configuration (this works reliably for frontend <-> backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Request model
class ChatRequest(BaseModel):
    message: str

# ✅ Main chatbot route
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message
    context_chunks = get_relevant_chunks(user_message)
    answer = get_gpt_response(user_message, context_chunks)
    return {"answer": answer}

# ✅ Simple GET route to confirm service is running
@app.get("/")
async def root():
    return {"message": "Hello from chatbot backend!"}
