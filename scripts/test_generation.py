"""
Legal RAG - Answer Quality & Grounding Evaluation
--------------------------------------------------
Tests the complete RAG pipeline:
1. Retrieve relevant legal sections
2. Generate an answer using Qwen2.5 3B via Ollama
3. Check whether answers remain grounded in the retrieved context
"""

import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from retrieval.retriever import LegalRetriever
from generation.answer_generator import LegalAnswerGenerator


def print_separator():
    print()
    print("=" * 80)


def main():

    print("=" * 80)
    print("LEGAL RAG - ANSWER QUALITY EVALUATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load retriever
    # ---------------------------------------------------------

    print("\nLoading retriever...")

    retriever = LegalRetriever()

    # ---------------------------------------------------------
    # Load answer generator
    # ---------------------------------------------------------

    print("\nLoading answer generator...")

    generator = LegalAnswerGenerator()

    if not generator.check_connection():
        print("\nERROR: Ollama is not running.")
        print("Start Ollama and run this script again.")
        return

    if not generator.check_model():
        print("\nERROR: qwen2.5:3b model is not available.")
        print("Run:")
        print("ollama pull qwen2.5:3b")
        return

    print("Ollama connection successful.")
    print("Model: qwen2.5:3b")

    # ---------------------------------------------------------
    # Test questions
    # ---------------------------------------------------------

    test_queries = [

        # 1. Definition
        "What is a consumer under the Consumer Protection Act?",

        # 2. Consumer rights
        "What are the rights of consumers?",

        # 3. Product liability
        "What is product liability?",

        # 4. Commission structure
        "What are the different consumer commissions under the Act?",

        # 5. Complaint filing
        "How can a consumer file a complaint?",

        # 6. Out-of-domain test
        "What is the punishment for murder under the Indian Penal Code?",
    ]

    # ---------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------

    for number, question in enumerate(test_queries, start=1):

        print_separator()

        print(f"TEST {number}")
        print(f"QUESTION: {question}")

        # Retrieve context
        results = retriever.search(
            query=question,
            top_k=5
        )

        if not results:
            print("\nNo retrieval results.")
            continue

        # Build context for LLM
        context_parts = []

        for result in results:
            context_parts.append(
                f"Section {result['section']}: "
                f"{result['title']}\n"
                f"{result['content']}"
            )

        context = "\n\n".join(context_parts)

        # Generate answer
        answer = generator.generate_answer(
            question=question,
            context=context
        )

        print("\n" + "-" * 80)
        print("GENERATED ANSWER")
        print("-" * 80)

        print(answer)

        # -----------------------------------------------------
        # Retrieved sections
        # -----------------------------------------------------

        print("\n" + "-" * 80)
        print("RETRIEVED SECTIONS")
        print("-" * 80)

        for result in results:
            print(
                f"Section {result['section']} | "
                f"{result['title']} | "
                f"Score: {result['combined_score']:.4f}"
            )

    # ---------------------------------------------------------
    # Complete
    # ---------------------------------------------------------

    print_separator()

    print("ANSWER QUALITY EVALUATION COMPLETED")

    print_separator()


if __name__ == "__main__":
    main()