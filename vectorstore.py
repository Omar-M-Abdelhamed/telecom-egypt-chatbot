"""
vectorstore.py
--------------
Manages the ChromaDB vector store:
  - Initialises (or loads) the persistent database from disk
  - Embeds text using sentence-transformers (free, runs locally)
  - Adds documents (from the scraper or uploaded files)
  - Searches for the most relevant chunks given a query

ChromaDB stores its data in ./chroma_db/ by default, so after the first
run the data persists and we never need to re-scrape.
"""

import os
import hashlib
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"          # where ChromaDB saves data
COLLECTION_NAME    = "telecom_egypt"        # logical bucket inside ChromaDB
EMBED_MODEL        = "paraphrase-multilingual-MiniLM-L12-v2"  # supports Arabic + English

# Text chunking settings
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 150   # overlap between adjacent chunks

# How many chunks to retrieve per query
TOP_K = 5


# ── Embedding function (sentence-transformers, runs locally) ──────────────────
def _get_embedding_function():
    """Return a ChromaDB-compatible embedding function using sentence-transformers."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


# ── ChromaDB client & collection ──────────────────────────────────────────────
_client: Optional[chromadb.ClientAPI] = None
_collection = None


def get_collection():
    """
    Return the ChromaDB collection, creating it (and the client) if needed.
    Data is persisted to CHROMA_PERSIST_DIR on disk.
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return _collection


# ── Check whether the DB already has data ─────────────────────────────────────
def is_populated() -> bool:
    """Return True if the collection already contains documents."""
    col = get_collection()
    return col.count() > 0


# ── Text chunking ──────────────────────────────────────────────────────────────
def _split_text(text: str) -> list[str]:
    """Split long text into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "،", " ", ""],  # Arabic comma included
    )
    return splitter.split_text(text)


def _make_id(source: str, chunk_index: int) -> str:
    """Create a stable, unique ID for a chunk (hash of source + index)."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Add documents ──────────────────────────────────────────────────────────────
def add_documents(pages: list[dict], verbose: bool = True) -> int:
    """
    Embed and store a list of pages in ChromaDB.

    Parameters
    ----------
    pages : list of {"url" | "filename": str, "text": str}
    verbose : print progress

    Returns
    -------
    int : number of chunks actually added
    """
    col = get_collection()
    total_added = 0

    for page in pages:
        source = page.get("url") or page.get("filename") or "unknown"
        text   = page.get("text", "").strip()

        if not text:
            continue

        chunks = _split_text(text)

        ids       = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = _make_id(source, i)

            # Skip chunks already stored (idempotent)
            existing = col.get(ids=[doc_id])
            if existing["ids"]:
                continue

            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({"source": source})

        if ids:
            col.add(ids=ids, documents=documents, metadatas=metadatas)
            total_added += len(ids)
            if verbose:
                print(f"[VectorStore] Added {len(ids)} chunks from: {source}")

    if verbose:
        print(f"[VectorStore] Total chunks added this session: {total_added}")
        print(f"[VectorStore] Collection now has {col.count()} chunks total.")

    return total_added


# ── Search ─────────────────────────────────────────────────────────────────────
def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Find the most relevant chunks for a user query.

    Returns
    -------
    list of {"text": str, "source": str, "distance": float}
    """
    col = get_collection()

    if col.count() == 0:
        return []

    results = col.query(
        query_texts=[query],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":     doc,
            "source":   meta.get("source", "unknown"),
            "distance": round(dist, 4),
        })

    return hits


# ── Delete a source ────────────────────────────────────────────────────────────
def delete_source(source: str) -> int:
    """Remove all chunks that came from a particular source URL or filename."""
    col = get_collection()
    results = col.get(where={"source": source})
    ids = results.get("ids", [])
    if ids:
        col.delete(ids=ids)
    return len(ids)
