import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from generation.grounding_validator import GroundingValidator


def run_test(
    validator,
    name,
    answer,
    results
):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    validation = validator.validate(
        answer,
        results
    )

    validator.display_validation(
        validation
    )


def main():

    validator = GroundingValidator()

    results = [
        {
            "section": "2",
            "title": "Definitions"
        },
        {
            "section": "18",
            "title": "Powers and functions of Central Authority"
        },
        {
            "section": "5",
            "title": "Objects of Central Council"
        }
    ]

    # ============================================================
    # TEST 1
    # Correct source mapping
    # ============================================================

    run_test(
        validator,
        "TEST 1 - CORRECT SOURCE MAPPING",

        """
Answer:
Consumer is defined in Section 2.

Relevant Sections:
Section 2

Explanation:
Section 2 contains the relevant definition.

Sources:
2
""",

        results
    )

    # ============================================================
    # TEST 2
    # Wrong section for SOURCE 1
    # ============================================================

    run_test(
        validator,
        "TEST 2 - SECTION/SOURCE MISMATCH",

        """
Answer:
The definition is provided in Section 7 of SOURCE 1.

Relevant Sections:
Section 7

Explanation:
Section 7 provides the definition.

Sources:
7
""",

        results
    )

    # ============================================================
    # TEST 3
    # Unsupported sections
    # ============================================================

    run_test(
        validator,
        "TEST 3 - UNSUPPORTED SECTION",

        """
Answer:
Consumer rights are provided in Sections 2, 9, 10 and 11.

Relevant Sections:
2, 9, 10 and 11

Explanation:
These sections contain consumer rights.

Sources:
2, 9, 10, 11
""",

        results
    )

    # ============================================================
    # TEST 4
    # Correct source references
    # ============================================================

    run_test(
        validator,
        "TEST 4 - SOURCE REFERENCES",

        """
Answer:
The relevant provision is in Section 2 of SOURCE 1.

Relevant Sections:
Section 2

Explanation:
SOURCE 1 contains Section 2.

Sources:
2
""",

        results
    )

    # ============================================================
    # TEST 5
    # Out of domain
    # ============================================================

    run_test(
        validator,
        "TEST 5 - OUT OF DOMAIN",

        """
Answer:
The retrieved legal context does not contain sufficient information to answer this question.

Relevant Sections:
Not applicable

Explanation:
The retrieved context does not contain information about the requested subject.

Sources:
Not applicable
""",

        results
    )


if __name__ == "__main__":
    main()