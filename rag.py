"""
rag.py
------
RAG pipeline with multi-turn conversation memory.
Uses the new google-genai SDK (google.genai) — the old google.generativeai is deprecated.

Flow per message:
  1. Retrieve top-K relevant chunks from ChromaDB (based on latest question)
  2. Inject retrieved context into the system instruction
  3. Send full conversation history + new question to Gemini via a chat session
  4. Stream the response back token by token
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

import vectorstore

load_dotenv()

# ── API key: Streamlit Cloud secrets first, then .env ─────────────────────────
def _get_api_key() -> str:
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")

GEMINI_API_KEY = _get_api_key()
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set.\n"
        "Create a .env file with:  GEMINI_API_KEY=your_key_here\n"
        "Get a free key at https://aistudio.google.com/app/apikey"
    )

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# ── System instruction ─────────────────────────────────────────────────────────
_BASE_SYSTEM = """You are a friendly and professional customer-service agent for Telecom Egypt (TE).

Rules:
- Always reply in the SAME language the customer used (Arabic, English, or Egyptian dialect).
- Be conversational and remember everything said earlier in this chat.
- Use the CONTEXT block below (when provided) to answer accurately.
- If the context doesn't have enough information, be honest and suggest the customer
  call Telecom Egypt support or visit te.eg directly.
- Never invent prices, speeds, or package names — only state what is in the context.
- Keep answers concise and helpful.
"""


def _build_system_with_context(chunks: list[dict]) -> str:
    """Prepend retrieved KB chunks to the system instruction for this turn."""
    if not chunks:
        return _BASE_SYSTEM
    context_parts = [
        f"[Source {i}: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ]
    context_block = "\n\n---\n\n".join(context_parts)
    return f"{_BASE_SYSTEM}\n\n=== RELEVANT CONTEXT ===\n{context_block}\n{'='*50}"


def _to_genai_history(messages: list[dict]) -> list[types.Content]:
    """
    Convert app.py message list → google.genai Content history.
    Excludes the last message (current user turn — sent separately).
    """
    history = []
    for msg in messages[:-1]:
        role = "model" if msg["role"] == "assistant" else "user"
        history.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )
    return history


def ask_stream(question: str, history: list[dict] = None, top_k: int = 5):
    """
    Stream a response from Gemini with full conversation memory.

    Yields
    ------
    (token: str, None)          while streaming
    (None, sources: list[str])  when done
    """
    # 1. Retrieve relevant chunks
    chunks  = vectorstore.search(question, top_k=top_k)
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    # 2. Build system instruction with context
    system_instruction = _build_system_with_context(chunks)

    # 3. Build generation config
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=1024,
    )

    # 4. Create chat session with prior history
    gemini_history = _to_genai_history(history or [])
    chat = client.chats.create(
        model=MODEL_NAME,
        config=config,
        history=gemini_history,
    )

    # 5. Stream the response
    for chunk in chat.send_message_stream(question):
        if chunk.text:
            yield chunk.text, None

    yield None, sources
