"""OpenAI response helper for the chatbot backend."""

import os
from typing import List

from openai import OpenAI

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.6-luna")
client = OpenAI()


def get_gpt_response(question: str, context_chunks: List[str]) -> str:
    """Generate an answer using optional retrieved context."""
    if context_chunks:
        context_text = "\n\n".join(context_chunks)
        instructions = (
            "You are a helpful assistant. Answer using the supplied context where possible. "
            "If the context does not contain the answer, say that clearly before using general knowledge."
        )
        user_input = f"Context:\n{context_text}\n\nQuestion: {question}"
    else:
        instructions = "You are a helpful assistant. Answer the user's question clearly and concisely."
        user_input = question

    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=user_input,
    )
    return response.output_text.strip()
