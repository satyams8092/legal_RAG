"""
Embedding and Vector Store Module
Consumer Protection Act, 2019

Step 6 of Legal RAG project.

Responsibilities:
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS
- Save metadata separately
- Load the persistent vector store later
"""

from pathlib import Path
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "cpa_2019.index"
METADATA_PATH = VECTORSTORE_DIR / "cpa_2019_metadata.json"


# ============================================================
# EMBEDDING MODEL
# ============================================================

_model = None


def get_embedding_model():
    """
    Load the embedding model only when required.
    """
    global _model

    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded successfully.")

    return _model


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(chunks):
    """
    Generate embeddings for a list of chunks.

    Each chunk must contain:
        - content

    Returns:
        numpy array of embeddings
    """

    if not chunks:
        raise ValueError("No chunks supplied for embedding.")

    texts = []

    for chunk in chunks:
        content = chunk.get("content", "").strip()

        if not content:
            raise ValueError(
                f"Empty content found in chunk: {chunk}"
            )

        texts.append(content)

    model = get_embedding_model()

    print()
    print("=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)

    print(f"Total chunks : {len(texts)}")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    print(f"Embedding shape : {embeddings.shape}")
    print(f"Embedding dimension : {embeddings.shape[1]}")

    return embeddings


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):
    """
    Create a FAISS cosine-similarity index.

    Since embeddings are normalized,
    inner product = cosine similarity.
    """

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings supplied.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# SAVE VECTOR STORE
# ============================================================

def save_vector_store(index, chunks):
    """
    Save FAISS index and chunk metadata.
    """

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # Save metadata
    metadata = []

    for chunk in chunks:
        metadata.append({
            "chunk_id": chunk.get("chunk_id"),
            "section_number": chunk.get("section_number"),
            "section_title": chunk.get("section_title"),
            "chapter": chunk.get("chapter"),
            "chunk_number": chunk.get("chunk_number"),
            "total_chunks": chunk.get("total_chunks"),
            "content": chunk.get("content")
        })

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)
    print("VECTOR STORE SAVED")
    print("=" * 70)

    print(f"FAISS index : {INDEX_PATH}")
    print(f"Metadata    : {METADATA_PATH}")
    print(f"Vectors     : {index.ntotal}")


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():
    """
    Load the persistent FAISS index and metadata.
    """

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found:\n{INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_PATH}"
        )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    return index, metadata


# ============================================================
# SEARCH
# ============================================================

def search_vector_store(query, top_k=5):
    """
    Search the persistent vector store.

    Returns the most relevant chunks.
    """

    index, metadata = load_vector_store()

    model = get_embedding_model()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        result = metadata[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results