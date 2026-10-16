from pathlib import Path
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# IMPORTS
# ============================================================

from ingestion.loader import extract_pdf_pages

from ingestion.parser import (
    find_act_start,
    parse_legal_sections,
    validate_sections,
    get_section,
)


# ============================================================
# PDF PATH
# ============================================================

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "indian_contract_act.pdf"
)


# ============================================================
# PRINT SECTION
# ============================================================

def print_section(section):

    print("\n" + "-" * 70)

    print(
        f"Section Number : "
        f"{section['section_number']}"
    )

    print(
        f"Section Title  : "
        f"{section['section_title']}"
    )

    print(
        f"Page Number    : "
        f"{section['page_number']}"
    )

    print(
        f"Content Length : "
        f"{len(section['content'])} characters"
    )

    print(
        f"Content        : "
        f"{section['content'][:700]}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 3: LEGAL CONTRACT ACT PARSER")
    print("=" * 70)

    # ========================================================
    # LOAD PDF
    # ========================================================

    pages = extract_pdf_pages(
        PDF_PATH
    )

    print(
        f"\nTotal pages extracted: "
        f"{len(pages)}"
    )

    # ========================================================
    # FIND ACT START
    # ========================================================

    start_index = find_act_start(
        pages
    )

    start_page = pages[
        start_index
    ]["page_number"]

    print(
        f"Actual Act starts at PDF page: "
        f"{start_page}"
    )

    # ========================================================
    # PARSE
    # ========================================================

    sections = parse_legal_sections(
        pages,
        start_page=start_page,
    )

    print(
        f"Total sections found: "
        f"{len(sections)}"
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    validation = validate_sections(
        sections
    )

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        f"\nTotal parsed sections : "
        f"{validation['total_sections']}"
    )

    print(
        f"First section         : "
        f"{validation['first_section']}"
    )

    print(
        f"Last section          : "
        f"{validation['last_section']}"
    )

    print(
        f"Duplicate sections    : "
        f"{validation['duplicates']}"
    )

    print(
        f"Missing test sections : "
        f"{validation['missing_required']}"
    )

    # ========================================================
    # FIRST 10
    # ========================================================

    print("\n" + "=" * 70)
    print("FIRST 10 PARSED SECTIONS")
    print("=" * 70)

    for section in sections[:10]:

        print_section(
            section
        )

    # ========================================================
    # IMPORTANT SECTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("IMPORTANT SECTION CHECK")
    print("=" * 70)

    important = [
        "1",
        "2",
        "3",
        "10",
        "16",
        "20",
        "23",
        "28",
        "56",
        "73",
        "74",
        "124",
        "148",
        "161",
        "178",
        "182",
    ]

    for number in important:

        section = get_section(
            sections,
            number,
        )

        if section:

            print(
                f"\n[FOUND] Section {number}"
            )

            print(
                f"Title: "
                f"{section['section_title']}"
            )

            print(
                f"Page: "
                f"{section['page_number']}"
            )

            print(
                f"Content: "
                f"{section['content'][:250]}"
            )

        else:

            print(
                f"\n[MISSING] Section {number}"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()