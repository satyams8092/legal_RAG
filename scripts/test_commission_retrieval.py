import os
import sys

# Memory-safe settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import LegalRetriever


INDEX_PATH = os.path.join(
    PROJECT_ROOT,
    "vectorstore",
    "cpa_2019.index"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "vectorstore",
    "cpa_2019_metadata.json"
)


def main():

    print("=" * 70)
    print("COMMISSION RETRIEVAL DIAGNOSTIC")
    print("=" * 70)

    retriever = LegalRetriever(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    queries = [
        "What are the different consumer commissions under the Act?",
        "What is the District Commission?",
        "What is the State Commission?",
        "What is the National Commission?",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = retriever.search(
            query=query,
            top_k=10
        )

        for i, result in enumerate(results, start=1):

            print("\n" + "-" * 70)

            print(f"Rank           : {i}")
            print(f"Section        : {result.get('section')}")
            print(f"Title          : {result.get('title')}")

            print(
                f"Semantic Score : "
                f"{result.get('semantic_score', result.get('score', 0)):.4f}"
            )

            print(
                f"Lexical Score  : "
                f"{result.get('lexical_score', 0):.4f}"
            )

            print(
                f"Combined Score : "
                f"{result.get('combined_score', 0):.4f}"
            )

            content = result.get("content", "")

            print("Content:")
            print(content[:500])


if __name__ == "__main__":
    main()