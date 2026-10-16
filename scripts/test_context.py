import sys
from pathlib import Path

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from retrieval.retriever import LegalRetriever
from generation.context_builder import build_context, build_sources


def main():

    print("=" * 70)
    print("STEP 8: LEGAL RAG CONTEXT BUILDER")
    print("=" * 70)

    # --------------------------------------------------
    # Load retriever
    # --------------------------------------------------

    print("\nLoading legal retriever...")

    retriever = LegalRetriever()

    print("Retriever loaded successfully.")

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    query = "What are the rights of consumers?"

    print("\nQUERY:")
    print(query)

    # --------------------------------------------------
    # Retrieve
    # --------------------------------------------------

    results = retriever.search(
        query=query,
        top_k=5
    )

    print("\nRetrieved results:", len(results))

    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context = build_context(results)

    print("\n")
    print("=" * 70)
    print("GENERATED CONTEXT")
    print("=" * 70)

    print(context)

    # --------------------------------------------------
    # Build sources
    # --------------------------------------------------

    sources = build_sources(results)

    print("\n")
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in sources:

        print(
            f"Section {source['section']} | "
            f"{source['title']} | "
            f"Score: {source['score']:.4f}"
        )

    print("\n")
    print("=" * 70)
    print("STEP 8 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()