import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from rag_pipeline import LegalRAGPipeline


def main():

    print("=" * 70)
    print("STEP 10: COMPLETE LEGAL RAG PIPELINE")
    print("=" * 70)

    pipeline = LegalRAGPipeline(
        top_k=5
    )

    question = "What are the rights of consumers?"

    print("\nQUESTION:")
    print(question)

    try:

        result = pipeline.ask(question)

        print("\n")
        print("=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(result["answer"])

        print("\n")
        print("=" * 70)
        print("SOURCES")
        print("=" * 70)

        for source in result["sources"]:

            print(
                f"Section {source['section']} | "
                f"{source['title']} | "
                f"Score: {source['score']:.4f}"
            )

    except RuntimeError as e:

        print("\n")
        print("=" * 70)
        print("GENERATION UNAVAILABLE")
        print("=" * 70)

        print(e)

        print("\nRetrieval portion is still working correctly.")

    print("\n")
    print("=" * 70)
    print("STEP 10 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()