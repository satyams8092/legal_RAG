"""
Legal Document Chunker
======================

Supports:
- Indian Contract Act parser output
- Consumer Protection Act 2019 JSON
- Different capitalization / naming of JSON fields

The chunker:
- Keeps sections separate
- Splits large sections
- Preserves legal metadata
- Creates embedding-ready text
"""

from typing import List, Dict, Any
import re


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
MIN_CONTENT_LENGTH = 20


# ============================================================
# FLEXIBLE FIELD READER
# ============================================================

def get_field(
    data: Dict[str, Any],
    *possible_names: str,
    default=None
):
    """
    Get a field regardless of capitalization or minor
    naming differences.

    Example:
        content
        Content
        CONTENT
        section_content
        Section Content
    """

    if not isinstance(data, dict):
        return default

    # Exact match first
    for name in possible_names:
        if name in data:
            return data[name]

    # Normalize keys
    normalized = {}

    for key in data.keys():

        normalized_key = re.sub(
            r"[^a-z0-9]",
            "",
            str(key).lower()
        )

        normalized[normalized_key] = key

    for name in possible_names:

        normalized_name = re.sub(
            r"[^a-z0-9]",
            "",
            str(name).lower()
        )

        if normalized_name in normalized:

            original_key = normalized[
                normalized_name
            ]

            return data[original_key]

    return default


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: Any) -> str:
    """
    Clean extracted legal text.
    """

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Normalize excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Remove spaces around newlines
    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    return text.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(text: str) -> List[str]:

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9(])",
        text
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# ============================================================
# LEGAL TEXT SPLITTING
# ============================================================

def split_legal_text(text: str) -> List[str]:
    """
    Split legal text into useful units.

    Priority:
    1. Paragraphs
    2. Numbered clauses
    3. Sentences
    """

    text = clean_text(text)

    if not text:
        return []

    # --------------------------------------------------------
    # Paragraphs
    # --------------------------------------------------------

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n{2,}",
            text
        )
        if p.strip()
    ]

    if len(paragraphs) > 1:
        return paragraphs

    # --------------------------------------------------------
    # Legal clauses
    # --------------------------------------------------------

    clauses = re.split(
        r"(?=\(\d+\)|\([a-z]\)|\([ivxlcdm]+\))",
        text,
        flags=re.IGNORECASE
    )

    clauses = [
        c.strip()
        for c in clauses
        if c.strip()
    ]

    if len(clauses) > 1:
        return clauses

    # --------------------------------------------------------
    # Sentences
    # --------------------------------------------------------

    return split_into_sentences(text)


# ============================================================
# SPLIT VERY LARGE TEXT
# ============================================================

def split_large_text(
    text: str,
    chunk_size: int,
    overlap: int
) -> List[str]:

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    pieces = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        piece = text[start:end]

        # Try to stop at whitespace
        if end < text_length:

            last_space = piece.rfind(" ")

            if last_space > chunk_size * 0.6:

                end = start + last_space

                piece = text[start:end]

        piece = piece.strip()

        if piece:
            pieces.append(piece)

        if end >= text_length:
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return pieces


# ============================================================
# EMBEDDING TEXT
# ============================================================

def build_embedding_text(
    act: str,
    section_number: Any,
    section_title: str,
    chapter: str,
    content: str
) -> str:
    """
    Build the final text that will be sent to the
    embedding model later.
    """

    parts = []

    if act:
        parts.append(
            f"Act: {clean_text(act)}"
        )

    if chapter:
        parts.append(
            f"Chapter: {clean_text(chapter)}"
        )

    if section_number is not None:
        parts.append(
            f"Section: {section_number}"
        )

    if section_title:
        parts.append(
            f"Section Title: {clean_text(section_title)}"
        )

    parts.append(
        f"Content: {clean_text(content)}"
    )

    return "\n".join(parts)


# ============================================================
# CHUNK ONE SECTION
# ============================================================

def chunk_section(
    section: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:

    if not isinstance(section, dict):
        raise TypeError(
            "Each section must be a dictionary."
        )

    # ========================================================
    # FLEXIBLE FIELD EXTRACTION
    # ========================================================

    act = get_field(
        section,
        "act",
        "act_name",
        "Act",
        "Act Name",
        default="Unknown Act"
    )

    section_number = get_field(
        section,
        "section_number",
        "section",
        "number",
        "Section Number",
        "Section",
        "Section No",
        default=None
    )

    section_title = get_field(
        section,
        "section_title",
        "title",
        "Section Title",
        "Title",
        default=""
    )

    chapter = get_field(
        section,
        "chapter",
        "Chapter",
        "chapter_name",
        "Chapter Name",
        default=""
    )

    page = get_field(
        section,
        "page",
        "page_number",
        "Page",
        "Page Number",
        default=None
    )

    content = get_field(
        section,
        "content",
        "Content",
        "section_content",
        "Section Content",
        "text",
        "Text",
        "body",
        "Body",
        default=""
    )

    # ========================================================
    # CLEAN VALUES
    # ========================================================

    act = clean_text(act)
    section_title = clean_text(section_title)
    chapter = clean_text(chapter)
    content = clean_text(content)

    # ========================================================
    # EMPTY CONTENT CHECK
    # ========================================================

    if len(content) < MIN_CONTENT_LENGTH:
        return []

    # ========================================================
    # SMALL SECTION
    # ========================================================

    if len(content) <= chunk_size:

        embedding_text = build_embedding_text(
            act,
            section_number,
            section_title,
            chapter,
            content
        )

        return [
            {
                "chunk_id":
                    f"{act}_section_{section_number}_chunk_1",

                "act":
                    act,

                "section_number":
                    str(section_number)
                    if section_number is not None
                    else "",

                "section_title":
                    section_title,

                "chapter":
                    chapter,

                "page":
                    page,

                "chunk_index":
                    1,

                "total_chunks":
                    1,

                "content":
                    content,

                "text":
                    embedding_text
            }
        ]

    # ========================================================
    # LARGE SECTION
    # ========================================================

    units = split_legal_text(content)

    chunks = []

    current = ""

    for unit in units:

        unit = unit.strip()

        if not unit:
            continue

        # ----------------------------------------------------
        # Unit fits current chunk
        # ----------------------------------------------------

        if (
            len(current)
            + len(unit)
            + 1
            <= chunk_size
        ):

            if current:
                current += " " + unit

            else:
                current = unit

        # ----------------------------------------------------
        # Unit does not fit
        # ----------------------------------------------------

        else:

            if current:

                chunks.append(
                    current.strip()
                )

            # ------------------------------------------------
            # Individual unit too large
            # ------------------------------------------------

            if len(unit) > chunk_size:

                pieces = split_large_text(
                    unit,
                    chunk_size,
                    chunk_overlap
                )

                if pieces:

                    chunks.extend(
                        pieces[:-1]
                    )

                    current = pieces[-1]

            else:

                current = unit

    # --------------------------------------------------------
    # Remaining content
    # --------------------------------------------------------

    if current:
        chunks.append(
            current.strip()
        )

    # ========================================================
    # ADD OVERLAP
    # ========================================================

    final_chunks = []

    for i, chunk in enumerate(chunks):

        if (
            i > 0
            and chunk_overlap > 0
        ):

            previous = chunks[i - 1]

            overlap_text = previous[
                -chunk_overlap:
            ]

            chunk = (
                overlap_text
                + " "
                + chunk
            )

        final_chunks.append(
            chunk.strip()
        )

    # ========================================================
    # CREATE RECORDS
    # ========================================================

    total_chunks = len(final_chunks)

    records = []

    for index, chunk in enumerate(
        final_chunks,
        start=1
    ):

        embedding_text = build_embedding_text(
            act,
            section_number,
            section_title,
            chapter,
            chunk
        )

        records.append(
            {
                "chunk_id":
                    f"{act}_section_{section_number}_chunk_{index}",

                "act":
                    act,

                "section_number":
                    str(section_number)
                    if section_number is not None
                    else "",

                "section_title":
                    section_title,

                "chapter":
                    chapter,

                "page":
                    page,

                "chunk_index":
                    index,

                "total_chunks":
                    total_chunks,

                "content":
                    chunk,

                "text":
                    embedding_text
            }
        )

    return records


# ============================================================
# CHUNK ALL SECTIONS
# ============================================================

def chunk_sections(
    sections: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:

    if not isinstance(sections, list):

        raise TypeError(
            "sections must be a list."
        )

    all_chunks = []

    for section in sections:

        section_chunks = chunk_section(
            section,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        all_chunks.extend(
            section_chunks
        )

    return all_chunks


# ============================================================
# VALIDATE CHUNKS
# ============================================================

def validate_chunks(
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:

    duplicate_ids = []
    empty_chunks = []
    missing_metadata = []

    seen_ids = set()

    for chunk in chunks:

        chunk_id = chunk.get(
            "chunk_id"
        )

        # ----------------------------------------------------
        # Duplicate IDs
        # ----------------------------------------------------

        if chunk_id in seen_ids:

            duplicate_ids.append(
                chunk_id
            )

        seen_ids.add(chunk_id)

        # ----------------------------------------------------
        # Empty content
        # ----------------------------------------------------

        if not chunk.get(
            "content",
            ""
        ).strip():

            empty_chunks.append(
                chunk_id
            )

        # ----------------------------------------------------
        # Required metadata
        # ----------------------------------------------------

        required_fields = [
            "chunk_id",
            "act",
            "section_number",
            "section_title",
            "content",
            "text"
        ]

        missing = [
            field
            for field in required_fields
            if not chunk.get(field)
        ]

        if missing:

            missing_metadata.append(
                {
                    "chunk_id":
                        chunk_id,

                    "missing":
                        missing
                }
            )

    return {
        "total_chunks":
            len(chunks),

        "duplicate_ids":
            duplicate_ids,

        "empty_chunks":
            empty_chunks,

        "missing_metadata":
            missing_metadata,

        "valid":
            (
                len(duplicate_ids) == 0
                and len(empty_chunks) == 0
                and len(missing_metadata) == 0
            )
    }


# ============================================================
# CHUNK SUMMARY
# ============================================================

def chunk_summary(
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not chunks:

        return {
            "total_chunks": 0,
            "min_length": 0,
            "max_length": 0,
            "average_length": 0
        }

    lengths = [
        len(
            chunk.get(
                "content",
                ""
            )
        )
        for chunk in chunks
    ]

    return {
        "total_chunks":
            len(chunks),

        "min_length":
            min(lengths),

        "max_length":
            max(lengths),

        "average_length":
            round(
                sum(lengths)
                / len(lengths),
                2
            )
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def create_chunks(
    sections: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:

    return chunk_sections(
        sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LEGAL DOCUMENT CHUNKER")
    print("=" * 70)

    test_section = {
        "Section Number": "1",
        "Section Title":
            "Short title, extent, commencement and application",
        "Chapter":
            "CHAPTER I — Preliminary",
        "Content":
            (
                "This Act may be called the Consumer Protection Act, 2019. "
                "It extends to the whole of India. "
                "It shall come into force on such date as the Central "
                "Government may, by notification, appoint."
            )
    }

    chunks = chunk_section(
        test_section
    )

    print(
        f"\nChunks generated: "
        f"{len(chunks)}"
    )

    for chunk in chunks:

        print("\n" + "-" * 70)

        print(
            f"Chunk ID       : "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Section        : "
            f"{chunk['section_number']}"
        )

        print(
            f"Title          : "
            f"{chunk['section_title']}"
        )

        print(
            f"Chapter        : "
            f"{chunk['chapter']}"
        )

        print(
            f"Content Length : "
            f"{len(chunk['content'])}"
        )

        print(
            f"Content        : "
            f"{chunk['content']}"
        )

    print("\n" + "=" * 70)

    validation = validate_chunks(
        chunks
    )

    print("VALIDATION")
    print("=" * 70)

    print(
        f"Total chunks  : "
        f"{validation['total_chunks']}"
    )

    print(
        f"Duplicate IDs : "
        f"{validation['duplicate_ids']}"
    )

    print(
        f"Empty chunks  : "
        f"{validation['empty_chunks']}"
    )

    print(
        f"Missing data  : "
        f"{validation['missing_metadata']}"
    )

    print(
        f"Valid         : "
        f"{validation['valid']}"
    )