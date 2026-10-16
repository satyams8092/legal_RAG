"""
STEP 6: CPA 2019 EMBEDDING TEST

Loads the validated CPA sections, creates the same chunks
using ingestion.chunker, generates embeddings, stores them
in FAISS, and tests semantic search.
"""

import sys
import json
from pathlib import Path


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from ingestion.chunker import chunk_sections
from ingestion.embedder import (
    generate_embeddings,
    create_faiss_index,
    save_vector_store,
    search_vector_store,
)


# ============================================================
# CONFIGURATION
# ============================================================

JSON_PATH = (
    BASE_DIR
    / "data"
    / "consumer_protection_act_2019.json"
)


# ============================================================
# LOAD CPA JSON
# ============================================================

def load_cpa_json():

    with open(
        JSON_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if isinstance(data, dict):

        if "sections" in data:
            return data["sections"]

        raise ValueError(
            "JSON dictionary does not contain 'sections'."
        )

    if isinstance(data, list):
        return data

    raise ValueError(
        "Unsupported JSON structure."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 6: CONSUMER PROTECTION ACT 2019 EMBEDDINGS")
    print("=" * 70)

    print()
    print(f"JSON file: {JSON_PATH}")

    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"CPA JSON file not found:\n{JSON_PATH}"
        )

    # --------------------------------------------------------
    # LOAD SECTIONS
    # --------------------------------------------------------

    sections = load_cpa_json()

    print(f"Sections loaded: {len(sections)}")

    if len(sections) != 107:
        print(
            f"[WARNING] Expected 107 sections, "
            f"but found {len(sections)}."
        )

    # --------------------------------------------------------
    # CREATE CHUNKS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING CHUNKS")
    print("=" * 70)

    chunks = chunk_sections(sections)

    print(f"Chunks generated: {len(chunks)}")

    if not chunks:
        raise ValueError(
            "No chunks were generated."
        )

    # --------------------------------------------------------
    # CHUNK VALIDATION
    # --------------------------------------------------------

    empty_chunks = []

    for chunk in chunks:

        content = chunk.get(
            "content",
            ""
        )

        if not content.strip():
            empty_chunks.append(
                chunk.get("chunk_id")
            )

    if empty_chunks:

        raise ValueError(
            f"Empty chunks found: {empty_chunks}"
        )

    print(
        f"Empty chunks: {len(empty_chunks)}"
    )

    # --------------------------------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------------------------------

    embeddings = generate_embeddings(
        chunks
    )

    # --------------------------------------------------------
    # VERIFY EMBEDDINGS
    # --------------------------------------------------------

    if len(embeddings) != len(chunks):

        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    print()
    print("=" * 70)
    print("EMBEDDING VALIDATION")
    print("=" * 70)

    print(
        f"Chunks     : {len(chunks)}"
    )

    print(
        f"Embeddings : {len(embeddings)}"
    )

    print(
        f"Dimension  : {embeddings.shape[1]}"
    )

    # --------------------------------------------------------
    # CREATE FAISS INDEX
    # --------------------------------------------------------

    index = create_faiss_index(
        embeddings
    )

    print()
    print("=" * 70)
    print("FAISS INDEX")
    print("=" * 70)

    print(
        f"Vectors stored: {index.ntotal}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_vector_store(
        index,
        chunks
    )

    # --------------------------------------------------------
    # SEMANTIC SEARCH TEST
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SEMANTIC SEARCH TEST")
    print("=" * 70)

    query = (
        "What are the rights of consumers "
        "under the Consumer Protection Act?"
    )

    print()
    print(f"Query: {query}")
    print()

    results = search_vector_store(
        query,
        top_k=5
    )

    if not results:

        print(
            "[WARNING] No search results returned."
        )

    else:

        for i, result in enumerate(
            results,
            start=1
        ):

            print("-" * 70)

            print(
                f"Result #{i}"
            )

            print(
                f"Score   : "
                f"{result.get('score', 0):.4f}"
            )

            print(
                f"Section : "
                f"{result.get('section_number')}"
            )

            print(
                f"Title   : "
                f"{result.get('section_title')}"
            )

            print(
                f"Chapter : "
                f"{result.get('chapter')}"
            )

            content = result.get(
                "content",
                ""
            )

            print(
                f"Content : {content[:500]}"
            )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("[SUCCESS] Step 6 completed.")
    print("[SUCCESS] CPA embeddings generated.")
    print("[SUCCESS] FAISS vector store created.")
    print("[SUCCESS] Metadata saved.")
    print("[SUCCESS] Semantic search tested.")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()