# gpt_logic.py

import openai
import os
from typing import List

# ✅ Load API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4"  # or use "gpt-3.5-turbo" to save tokens

def get_gpt_response(question: str, context_chunks: List[str]) -> str:
    if context_chunks:
        context_text = "\n\n".join(context_chunks)
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the context is unclear or insufficient, use general knowledge."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
    else:
        system_prompt = "You are a helpful assistant. Answer the user's question."
        user_prompt = question

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

