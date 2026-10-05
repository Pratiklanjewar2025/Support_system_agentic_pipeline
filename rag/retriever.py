# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
RAG (Retrieval-Augmented Generation) module.
Loads documents, creates FAISS vector index, and retrieves relevant chunks.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent.parent / "documents"
INDEX_PATH = Path(__file__).parent.parent / "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4

# Singleton model (loaded once)
_model: SentenceTransformer = None
_index: faiss.Index = None
_chunks: List[dict] = None  # [{text, source}]


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _chunk_text(text: str, source: str) -> List[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk.strip(), "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _load_documents() -> List[dict]:
    """Load all .md files from the documents directory."""
    all_chunks = []
    for md_file in sorted(DOCS_DIR.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        chunks = _chunk_text(content, source=md_file.name)
        all_chunks.extend(chunks)
        print(f"  [doc] Loaded '{md_file.name}' -> {len(chunks)} chunks")
    return all_chunks


def build_index() -> None:
    """Build FAISS index from documents and save to disk."""
    print("\n[*] Building FAISS index from knowledge base...")
    chunks = _load_documents()
    if not chunks:
        raise FileNotFoundError(f"No .md documents found in {DOCS_DIR}")

    model = _get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # L2 index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    INDEX_PATH.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH / "index.faiss"))
    with open(INDEX_PATH / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"[OK] Index built: {len(chunks)} total chunks stored.\n")


def load_index() -> Tuple[faiss.Index, List[dict]]:
    """Load FAISS index from disk (build if not present)."""
    global _index, _chunks
    if _index is not None and _chunks is not None:
        return _index, _chunks

    index_file = INDEX_PATH / "index.faiss"
    chunks_file = INDEX_PATH / "chunks.pkl"

    if not index_file.exists() or not chunks_file.exists():
        build_index()

    _index = faiss.read_index(str(index_file))
    with open(chunks_file, "rb") as f:
        _chunks = pickle.load(f)

    return _index, _chunks


def retrieve(query: str, top_k: int = TOP_K) -> List[dict]:
    """
    Retrieve the most relevant document chunks for a query.
    Returns list of dicts: [{text, source, score}]
    """
    index, chunks = load_index()
    model = _get_model()

    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append({
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "score": float(dist),
            })
    return results
