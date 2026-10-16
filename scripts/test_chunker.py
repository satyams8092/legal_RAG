import json
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunker import (
    chunk_sections,
    validate_chunks,
    chunk_summary
)


# ============================================================
# CPA 2019 JSON
# ============================================================

JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "consumer_protection_act_2019.json"
)


print("=" * 70)
print("STEP 5: CONSUMER PROTECTION ACT 2019 CHUNKER")
print("=" * 70)

print(f"\nJSON file: {JSON_PATH}")


# ============================================================
# LOAD JSON
# ============================================================

with open(
    JSON_PATH,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


# ============================================================
# HANDLE JSON FORMAT
# ============================================================

if isinstance(data, list):

    sections = data

elif isinstance(data, dict):

    if "sections" in data:
        sections = data["sections"]

    elif "records" in data:
        sections = data["records"]

    else:
        raise ValueError(
            "Could not find sections/records in JSON."
        )

else:

    raise ValueError(
        "Unsupported JSON structure."
    )


print(f"Sections loaded: {len(sections)}")


# ============================================================
# ADD ACT NAME
# ============================================================

for section in sections:

    section.setdefault(
        "act",
        "Consumer Protection Act, 2019"
    )


# ============================================================
# CHUNK
# ============================================================

chunks = chunk_sections(
    sections,
    chunk_size=1500,
    chunk_overlap=200
)


# ============================================================
# SUMMARY
# ============================================================

summary = chunk_summary(chunks)

print("\n" + "=" * 70)
print("CHUNKING SUMMARY")
print("=" * 70)

print(
    f"Total sections : {len(sections)}"
)

print(
    f"Total chunks   : {summary['total_chunks']}"
)

print(
    f"Minimum length : {summary['min_length']}"
)

print(
    f"Maximum length : {summary['max_length']}"
)

print(
    f"Average length : {summary['average_length']}"
)


# ============================================================
# VALIDATION
# ============================================================

validation = validate_chunks(chunks)

print("\n" + "=" * 70)
print("CHUNK VALIDATION")
print("=" * 70)

print(
    f"Total chunks       : "
    f"{validation['total_chunks']}"
)

print(
    f"Duplicate IDs      : "
    f"{validation['duplicate_ids']}"
)

print(
    f"Empty chunks       : "
    f"{validation['empty_chunks']}"
)

print(
    f"Missing metadata   : "
    f"{validation['missing_metadata']}"
)

print(
    f"Validation status  : "
    f"{'PASS' if validation['valid'] else 'FAIL'}"
)


# ============================================================
# FIRST 10 CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CHUNKS")
print("=" * 70)

for chunk in chunks[:10]:

    print("\n" + "-" * 70)

    print(
        f"Chunk ID       : {chunk['chunk_id']}"
    )

    print(
        f"Section        : {chunk['section_number']}"
    )

    print(
        f"Title          : {chunk['section_title']}"
    )

    print(
        f"Chapter        : {chunk['chapter']}"
    )

    print(
        f"Chunk          : "
        f"{chunk['chunk_index']}/"
        f"{chunk['total_chunks']}"
    )

    print(
        f"Content Length : "
        f"{len(chunk['content'])}"
    )

    print(
        f"Content        : "
        f"{chunk['content'][:500]}"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

if validation["valid"]:

    print(
        "[SUCCESS] Chunker validation passed."
    )

    print(
        "[SUCCESS] Ready for embedding stage."
    )

else:

    print(
        "[ERROR] Chunk validation failed."
    )

print("=" * 70)