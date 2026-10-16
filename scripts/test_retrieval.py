# scripts/test_retrieval.py

import os
import sys

# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from retrieval.retriever import LegalRetriever


# ============================================================
# Configuration
# ============================================================

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


# ============================================================
# Test cases
# ============================================================

TEST_CASES = [

    {
        "name": "Consumer Definition",
        "query": "What is a consumer under the Consumer Protection Act?",
        "expected_sections": ["2"]
    },

    {
        "name": "Consumer Rights",
        "query": "What are the rights of consumers?",
        "expected_sections": ["2"]
    },

    {
        "name": "Product Liability",
        "query": "What is product liability?",
        "expected_sections": ["83"]
    },

    {
        "name": "Consumer Commission Structure",
        "query": "What are the different consumer commissions under the Act?",
        "expected_sections": ["28", "42", "53"]
    },

    {
        "name": "Complaint Filing",
        "query": "How can a consumer file a complaint?",
        "expected_sections": ["35", "38"]
    }

]


# ============================================================
# Helper function
# ============================================================

def section_found(
    expected_sections,
    retrieved_sections
):

    return all(
        section in retrieved_sections
        for section in expected_sections
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("LEGAL RAG - RETRIEVAL EVALUATION")
    print("=" * 80)

    print()
    print("Initializing retriever...")

    retriever = LegalRetriever(
        INDEX_PATH,
        METADATA_PATH
    )

    print()
    print("Retriever initialized successfully.")
    print()

    total_tests = len(TEST_CASES)
    passed_tests = 0

    # ========================================================
    # Run tests
    # ========================================================

    for test_number, test in enumerate(
        TEST_CASES,
        start=1
    ):

        print("=" * 80)
        print(
            f"TEST {test_number}/{total_tests}"
        )
        print("=" * 80)

        print()
        print("Query:")
        print(test["query"])

        print()
        print("Expected important sections:")
        print(
            ", ".join(
                test["expected_sections"]
            )
        )

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        results = retriever.search(
            test["query"],
            top_k=5
        )

        # ----------------------------------------------------
        # Extract retrieved sections
        # ----------------------------------------------------

        retrieved_sections = [
            str(result["section_number"])
            for result in results
        ]

        print()
        print("Retrieved sections:")

        for rank, result in enumerate(
            results,
            start=1
        ):

            section_number = result[
                "section_number"
            ]

            section_title = result[
                "section_title"
            ]

            semantic_score = result[
                "semantic_score"
            ]

            lexical_score = result[
                "lexical_score"
            ]

            structural_score = result[
                "structural_score"
            ]

            combined_score = result[
                "combined_score"
            ]

            print(
                f"  {rank}. "
                f"Section {section_number} - "
                f"{section_title}"
            )

            print(
                f"     Semantic   : "
                f"{semantic_score:.4f}"
            )

            print(
                f"     Lexical    : "
                f"{lexical_score:.4f}"
            )

            print(
                f"     Structural : "
                f"{structural_score:.4f}"
            )

            print(
                f"     Combined   : "
                f"{combined_score:.4f}"
            )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        passed = section_found(
            test["expected_sections"],
            retrieved_sections
        )

        print()

        if passed:

            print("RESULT: PASS")
            passed_tests += 1

        else:

            print("RESULT: FAIL")

            missing = [
                section
                for section in test["expected_sections"]
                if section not in retrieved_sections
            ]

            print()
            print(
                "Missing expected sections:"
            )

            print(
                ", ".join(missing)
            )

        print()

    # ========================================================
    # Final summary
    # ========================================================

    failed_tests = (
        total_tests - passed_tests
    )

    accuracy = (
        passed_tests / total_tests
    ) * 100

    print("=" * 80)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80)

    print()
    print(
        f"Passed: {passed_tests}/{total_tests}"
    )

    print(
        f"Failed: {failed_tests}/{total_tests}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print()

    if passed_tests == total_tests:

        print(
            "SUCCESS: All retrieval tests passed."
        )

    else:

        print(
            "Some retrieval tests failed."
        )

    print()
    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()