# scripts/test_claim_grounding.py

import os
import sys


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from sentence_transformers import SentenceTransformer

from retrieval.retriever import LegalRetriever

from generation.claim_grounding import (
    ClaimGroundingValidator
)


# ============================================================
# Paths
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
# Main
# ============================================================

def main():

    print("=" * 80)
    print("LEGAL RAG - CLAIM LEVEL GROUNDING TEST")
    print("=" * 80)

    # ========================================================
    # Load retriever
    # ========================================================

    print()
    print("Loading retriever...")

    retriever = LegalRetriever(
        INDEX_PATH,
        METADATA_PATH
    )

    # ========================================================
    # Create validator using the SAME embedding model
    # ========================================================

    print()
    print("Creating claim grounding validator...")

    validator = ClaimGroundingValidator(
        retriever.model
    )

    print(
        "Claim grounding validator initialized."
    )

    # ========================================================
    # Test question
    # ========================================================

    question = (
        "What is a consumer under the "
        "Consumer Protection Act?"
    )

    print()
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(question)

    # ========================================================
    # Retrieve context
    # ========================================================

    results = retriever.search(
        question,
        top_k=5
    )

    print()
    print("=" * 80)
    print("RETRIEVED SECTIONS")
    print("=" * 80)

    for result in results:

        print(
            f"Section "
            f"{result['section_number']} - "
            f"{result['section_title']}"
        )

    # ========================================================
    # Test answer 1
    # ========================================================

    good_answer = """
Answer:

A consumer is a person who buys goods or hires or avails
services for consideration, subject to the conditions
provided in the Act.

Relevant Sections:

Section 2

Explanation:

Section 2 contains the definitions under the Consumer
Protection Act, including the definition of consumer.

Sources:

Section 2
"""

    print()
    print("=" * 80)
    print("TEST 1 - SUPPORTED ANSWER")
    print("=" * 80)

    validation = validator.validate(
        good_answer,
        results
    )

    validator.display_validation(
        validation
    )

    # ========================================================
    # Test answer 2
    # ========================================================

    bad_answer = """
Answer:

A consumer is a person who buys goods exclusively for
resale and commercial purposes.

Relevant Sections:

Section 7

Explanation:

The Act defines a consumer as a person who purchases
goods for resale.

Sources:

Section 7
"""

    print()
    print("=" * 80)
    print("TEST 2 - POTENTIALLY UNSUPPORTED ANSWER")
    print("=" * 80)

    validation = validator.validate(
        bad_answer,
        results
    )

    validator.display_validation(
        validation
    )

    # ========================================================
    # Test answer 3
    # ========================================================

    mixed_answer = """
Answer:

A consumer is a person who buys goods or hires services
for consideration under the Act.

The Consumer Protection Act was enacted by the Parliament
of India.

The Act provides consumer rights and protections.

Relevant Sections:

Section 2

Sources:

Section 2
"""

    print()
    print("=" * 80)
    print("TEST 3 - MIXED ANSWER")
    print("=" * 80)

    validation = validator.validate(
        mixed_answer,
        results
    )

    validator.display_validation(
        validation
    )

    print()
    print("=" * 80)
    print("CLAIM GROUNDING TEST COMPLETED")
    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()