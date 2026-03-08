"""
rag.py
------
RAG pipeline with multi-turn conversation memory.

Flow per message:
  1. Retrieve top-K relevant chunks from ChromaDB (based on latest question)
  2. Inject retrieved context into the system instruction
  3. Send full conversation history + new question to Gemini via a chat session
  4. Stream the response back token by token
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

import vectorstore

load_dotenv()

# Support both Streamlit Cloud secrets (for deployment) and .env (for local dev)
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

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

# Base system instruction (no context yet — context is injected per turn)
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

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    return f"{_BASE_SYSTEM}\n\n=== RELEVANT CONTEXT FROM KNOWLEDGE BASE ===\n{context_block}\n{'='*50}"


def _to_gemini_history(messages: list[dict]) -> list[dict]:
    """
    Convert app.py message list → Gemini chat history format.
    Gemini roles: "user" | "model"
    Skips the last message (current user turn — sent separately).
    """
    history = []
    # All messages except the last one (which is the current user question)
    for msg in messages[:-1]:
        role = "model" if msg["role"] == "assistant" else "user"
        history.append({
            "role": role,
            "parts": [{"text": msg["content"]}],
        })
    return history


def ask_stream(question: str, history: list[dict] = None, top_k: int = 5):
    """
    Stream a response from Gemini, with full conversation memory.

    Parameters
    ----------
    question : the latest user message
    history  : full message list from st.session_state.messages
               (list of {"role": "user"|"assistant", "content": str})
    top_k    : number of KB chunks to retrieve

    Yields
    ------
    (token: str, None)  while streaming
    (None, sources: list[str])  when done
    """
    # 1. Retrieve context relevant to the latest question
    chunks = vectorstore.search(question, top_k=top_k)
    sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

    # 2. Build system instruction with injected context
    system_instruction = _build_system_with_context(chunks)

    # 3. Build Gemini chat history from previous turns
    gemini_history = _to_gemini_history(history or [])

    # 4. Create a fresh chat session with the full history
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction,
        generation_config={
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 1024,
        },
    )
    chat = model.start_chat(history=gemini_history)

    # 5. Send the current question and stream the response
    response = chat.send_message(question, stream=True)
    for part in response:
        if part.text:
            yield part.text, None

    yield None, sources
