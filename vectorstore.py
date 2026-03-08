"""
vectorstore.py
--------------
Pure numpy + sentence-transformers vector store.
No ChromaDB — eliminates all pydantic/chromadb compatibility issues.

Data is persisted to ./vector_store.pkl on disk so scraping only happens once.
"""

import os
import pickle
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ──────────────────────────────────────────────────────────────
PERSIST_PATH = "./vector_store.pkl"
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"  # Arabic + English
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K         = 5

# ── In-memory state ────────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None

# Store shape: {"ids": list, "embeddings": np.ndarray | None, "documents": list, "metadatas": list}
_store: dict | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _load_store() -> dict:
    global _store
    if _store is not None:
        return _store
    if os.path.exists(PERSIST_PATH):
        with open(PERSIST_PATH, "rb") as f:
            _store = pickle.load(f)
    else:
        _store = {"ids": [], "embeddings": None, "documents": [], "metadatas": []}
    return _store


def _save_store() -> None:
    with open(PERSIST_PATH, "wb") as f:
        pickle.dump(_store, f)


# ── Public API ─────────────────────────────────────────────────────────────────

def is_populated() -> bool:
    return len(_load_store()["ids"]) > 0


def _split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "،", " ", ""],
    )
    return splitter.split_text(text)


def _make_id(source: str, index: int) -> str:
    return hashlib.md5(f"{source}::{index}".encode()).hexdigest()


def add_documents(pages: list[dict], verbose: bool = True) -> int:
    """Embed and persist a list of {"url"|"filename": str, "text": str} dicts."""
    store = _load_store()
    model = _get_model()
    existing_ids = set(store["ids"])

    new_ids, new_docs, new_metas = [], [], []

    for page in pages:
        source = page.get("url") or page.get("filename") or "unknown"
        text   = page.get("text", "").strip()
        if not text:
            continue
        for i, chunk in enumerate(_split_text(text)):
            doc_id = _make_id(source, i)
            if doc_id in existing_ids:
                continue
            new_ids.append(doc_id)
            new_docs.append(chunk)
            new_metas.append({"source": source})

    if new_ids:
        embeddings = model.encode(new_docs, show_progress_bar=verbose, batch_size=32)

        store["ids"].extend(new_ids)
        store["documents"].extend(new_docs)
        store["metadatas"].extend(new_metas)
        store["embeddings"] = (
            embeddings if store["embeddings"] is None
            else np.vstack([store["embeddings"], embeddings])
        )
        _save_store()

        if verbose:
            print(f"[VectorStore] Added {len(new_ids)} chunks. Total: {len(store['ids'])}")

    return len(new_ids)


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return top-k most similar chunks to the query using cosine similarity."""
    store = _load_store()
    if not store["ids"]:
        return []

    model = _get_model()
    q_emb = model.encode([query])                          # shape (1, dim)

    embs  = store["embeddings"]                            # shape (n, dim)
    # Normalise rows for cosine similarity
    embs_norm = embs  / (np.linalg.norm(embs,  axis=1, keepdims=True) + 1e-10)
    q_norm    = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)

    scores = (embs_norm @ q_norm.T).flatten()
    top_idx = np.argsort(scores)[::-1][:min(top_k, len(scores))]

    return [
        {
            "text":     store["documents"][i],
            "source":   store["metadatas"][i]["source"],
            "distance": float(1 - scores[i]),
        }
        for i in top_idx
    ]


def get_collection():
    """Thin shim so app.py chunk-count display still works."""
    class _Compat:
        def count(self):
            return len(_load_store()["ids"])
    return _Compat()
