from pathlib import Path
import sys

# Allow importing from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.json_loader import LegalJSONLoader


# ============================================================
# CONFIGURATION
# ============================================================

JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "consumer_protection_act_2019.json"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 4: CONSUMER PROTECTION ACT 2019 JSON LOADER")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

loader = LegalJSONLoader(JSON_PATH)

records, errors = loader.load_and_prepare()


# ============================================================
# BASIC INFORMATION
# ============================================================

print()
print(f"JSON file          : {JSON_PATH}")
print(f"Valid records      : {len(records)}")
print(f"Validation errors  : {len(errors)}")


# ============================================================
# ERRORS
# ============================================================

if errors:

    print()
    print("=" * 70)
    print("VALIDATION ERRORS")
    print("=" * 70)

    for error in errors:
        print(f"[ERROR] {error}")

else:

    print()
    print("[OK] No validation errors found.")


# ============================================================
# SECTION ANALYSIS
# ============================================================

section_numbers = []

for record in records:

    try:
        section_number = int(record["section"])
        section_numbers.append(section_number)

    except ValueError:
        print(
            f"[WARNING] Non-numeric section: "
            f"{record['section']}"
        )


# Remove duplicates for analysis
unique_sections = sorted(set(section_numbers))

duplicates = sorted(
    {
        section
        for section in section_numbers
        if section_numbers.count(section) > 1
    }
)


# ============================================================
# SECTION RANGE
# ============================================================

print()
print("=" * 70)
print("SECTION ANALYSIS")
print("=" * 70)

if unique_sections:

    print(f"First section      : {unique_sections[0]}")
    print(f"Last section       : {unique_sections[-1]}")
    print(f"Unique sections    : {len(unique_sections)}")

else:

    print("No numeric sections found.")


print(
    f"Duplicate sections : "
    f"{duplicates if duplicates else 'None'}"
)


# ============================================================
# FIRST 10 SECTIONS
# ============================================================

print()
print("=" * 70)
print("FIRST 10 CPA SECTIONS")
print("=" * 70)

for record in records[:10]:

    print()
    print("-" * 70)

    print(
        f"Section Number : {record['section']}"
    )

    print(
        f"Section Title  : {record['title']}"
    )

    print(
        f"Chapter        : {record.get('chapter')}"
    )

    print(
        f"Content Length : {len(record['text'])} characters"
    )

    preview = record["text"][:500]

    print(
        f"Content        : {preview}"
    )


# ============================================================
# IMPORTANT SECTION CHECK
# ============================================================

IMPORTANT_SECTIONS = [
    "2",
    "3",
    "10",
    "17",
    "20",
    "35",
    "38",
    "69",
    "75",
    "88",
    "89",
    "90",
    "96",
]


print()
print("=" * 70)
print("IMPORTANT CPA SECTION CHECK")
print("=" * 70)

record_map = {
    str(record["section"]): record
    for record in records
}


for section in IMPORTANT_SECTIONS:

    record = record_map.get(section)

    if record:

        print()
        print(
            f"[FOUND] Section {section}"
        )

        print(
            f"Title: {record['title']}"
        )

        print(
            f"Chapter: {record.get('chapter')}"
        )

        print(
            f"Content: {record['text'][:350]}"
        )

    else:

        print(
            f"[MISSING] Section {section}"
        )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("FINAL STATUS")
print("=" * 70)

if records and not errors:

    print("[SUCCESS] CPA JSON loaded successfully.")
    print(
        "[SUCCESS] Ready for the chunking/embedding stage."
    )

else:

    print(
        "[FAILED] Fix the validation errors before proceeding."
    )