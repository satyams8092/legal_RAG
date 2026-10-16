from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ============================================================
# SECTION NUMBER PATTERN
# ============================================================

SECTION_NUMBER_PATTERN = r"\d+[A-Z]?"


# ============================================================
# KNOWN FOOTNOTE / EDITORIAL PATTERNS
# ============================================================

FOOTNOTE_START_PATTERN = re.compile(
    r"^\d+\.\s+"
    r"(For|The words|The word|As to|Ins\.|Rep\.|Subs\.|"
    r"S\.|See|Added|Inserted|Omitted|Words|Substituted|"
    r"Provided|Vide|Act|Section|In|By|This)",
    re.IGNORECASE,
)


# ============================================================
# NORMALIZE SECTION NUMBER
# ============================================================

def normalize_section_number(raw_number: str) -> str:
    """
    Normalize section numbers affected by PDF extraction.

    Examples:
        16      -> 16
        19A     -> 19A
        1[16    -> 16
        2[19A   -> 19A
        3[178   -> 178
        1151    -> 151
        1161    -> 161
    """

    raw_number = raw_number.strip()

    # --------------------------------------------------------
    # Footnote prefix merged with section number
    # --------------------------------------------------------

    if "[" in raw_number:
        value = raw_number.split("[", 1)[1].strip()

        if re.fullmatch(
            SECTION_NUMBER_PATTERN,
            value
        ):
            return value

    # --------------------------------------------------------
    # Known merged section numbers
    # --------------------------------------------------------

    merged_numbers = {
        "1151": "151",
        "1161": "161",
    }

    if raw_number in merged_numbers:
        return merged_numbers[raw_number]

    return raw_number


# ============================================================
# FIND ACT START
# ============================================================

def find_act_start(pages: List[Dict]) -> int:
    """
    Find the zero-based index of the page where the actual
    Indian Contract Act begins.

    The actual Act starts where Section 1 begins.
    """

    # --------------------------------------------------------
    # First preference:
    # Search for the actual opening sentence of Section 1.
    # --------------------------------------------------------

    for index, page in enumerate(pages):

        text = page.get("text", "")

        if re.search(
            r"This Act may be called\s+the Indian Contract Act,\s*1872",
            text,
            re.IGNORECASE,
        ):
            return index

    # --------------------------------------------------------
    # Fallback:
    # Search for "An Act" followed by Indian Contract Act.
    # --------------------------------------------------------

    for index, page in enumerate(pages):

        text = page.get("text", "")

        if re.search(
            r"An Act.*Indian Contract Act",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            return index

    return 0


# ============================================================
# CHECK WHETHER LINE IS OBVIOUS FOOTNOTE
# ============================================================

def is_obvious_footnote(line: str) -> bool:
    """
    Detect obvious editorial / PDF footnote lines.

    Important:
    This prevents lines such as:

        1. For the Statement of Objects and Reasons...
        2. The words...
        3. The words...

    from being incorrectly parsed as Sections 1, 2 and 3.
    """

    if not line:
        return False

    stripped = line.strip()

    if FOOTNOTE_START_PATTERN.match(stripped):
        return True

    # Common amendment/editorial wording.
    editorial_patterns = [
        r"^\d+\.\s+for the statement",
        r"^\d+\.\s+the words",
        r"^\d+\.\s+the word",
        r"^\d+\.\s+inserted",
        r"^\d+\.\s+substituted",
        r"^\d+\.\s+omitted",
        r"^\d+\.\s+added",
        r"^\d+\.\s+rep\.",
        r"^\d+\.\s+subs\.",
        r"^\d+\.\s+ins\.",
        r"^\d+\.\s+see",
        r"^\d+\.\s+vide",
    ]

    for pattern in editorial_patterns:

        if re.match(
            pattern,
            stripped,
            re.IGNORECASE,
        ):
            return True

    return False


# ============================================================
# SECTION HEADING
# ============================================================

def find_section_heading(
    line: str,
) -> Optional[Tuple[str, str]]:
    """
    Detect a genuine legal section heading.

    Examples:

        1. Short title.—This Act may be called...
        16. “Undue influence” defined.—...
        19A. Power to set aside contract...
        1[16. Undue influence...
        2[19A. Power to...

    Returns:

        (section_number, section_title)

    or None.
    """

    if not line:
        return None

    original = line.strip()

    if not original:
        return None

    # --------------------------------------------------------
    # IMPORTANT:
    # Reject obvious PDF footnotes BEFORE section detection.
    # --------------------------------------------------------

    if is_obvious_footnote(original):
        return None

    # --------------------------------------------------------
    # Remove PDF footnote prefix.
    #
    # Examples:
    #   1[16. ...
    #   2[19A. ...
    #   3[178. ...
    # --------------------------------------------------------

    cleaned = re.sub(
        r"^\d+\[\s*",
        "",
        original,
    ).strip()

    # --------------------------------------------------------
    # Standard section heading
    # --------------------------------------------------------

    match = re.match(
        rf"^({SECTION_NUMBER_PATTERN})\.\s+(.+)$",
        cleaned,
    )

    if not match:
        return None

    raw_number = match.group(1)
    title = match.group(2).strip()

    # --------------------------------------------------------
    # Additional protection against footnotes.
    # --------------------------------------------------------

    if is_obvious_footnote(
        f"{raw_number}. {title}"
    ):
        return None

    section_number = normalize_section_number(
        raw_number
    )

    if not section_number:
        return None

    # --------------------------------------------------------
    # A genuine section heading should contain a title.
    # --------------------------------------------------------

    if len(title) < 2:
        return None

    return section_number, title


# ============================================================
# CLEAN PAGE TEXT
# ============================================================

def clean_page_text(text: str) -> str:
    """
    Basic PDF text cleanup.
    """

    if not text:
        return ""

    lines = text.splitlines()

    cleaned = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # ----------------------------------------------------
        # Standalone page numbers
        # ----------------------------------------------------

        if re.fullmatch(r"\d+", line):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ============================================================
# REMOVE OBVIOUS FOOTNOTES
# ============================================================

def remove_obvious_footnotes(text: str) -> str:
    """
    Remove obvious PDF footnote lines while preserving
    legitimate numbered clauses such as:

        (1)
        (2)
        (a)
        (b)

    and normal legal content.
    """

    if not text:
        return ""

    lines = text.splitlines()

    cleaned = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        if is_obvious_footnote(stripped):
            continue

        cleaned.append(stripped)

    return " ".join(cleaned)


# ============================================================
# CLEAN SECTION CONTENT
# ============================================================

def clean_section_content(text: str) -> str:
    """
    Clean extracted legal section content without destroying
    numbered legal clauses.
    """

    if not text:
        return ""

    # --------------------------------------------------------
    # Remove standalone PDF page numbers.
    # --------------------------------------------------------

    text = re.sub(
        r"\s+\d+\s+",
        " ",
        text,
    )

    # --------------------------------------------------------
    # Remove obvious footnote blocks.
    # --------------------------------------------------------

    text = remove_obvious_footnotes(text)

    # --------------------------------------------------------
    # Fix excessive whitespace.
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# PARSE LEGAL SECTIONS
# ============================================================

def parse_legal_sections(
    pages: List[Dict],
    start_page: Optional[int] = None,
) -> List[Dict]:
    """
    Parse the Indian Contract Act into individual sections.

    Parameters
    ----------
    pages:
        List returned by extract_pdf_pages().

    start_page:
        Actual PDF page number where the Act begins.
        This is 1-based.

        Example:
            start_page=10

    Returns
    -------
    List[Dict]

    Each section contains:

        section_number
        section_title
        page_number
        content
    """

    # --------------------------------------------------------
    # Determine starting page.
    # --------------------------------------------------------

    if start_page is not None:

        # start_page is a PDF page number.
        start_index = max(
            0,
            start_page - 1,
        )

    else:

        # find_act_start returns zero-based index.
        start_index = find_act_start(pages)

    sections: List[Dict] = []

    current_section: Optional[Dict] = None

    # ========================================================
    # PROCESS PAGES
    # ========================================================

    for page_index in range(
        start_index,
        len(pages),
    ):

        page = pages[page_index]

        page_number = page.get(
            "page_number",
            page_index + 1,
        )

        raw_text = page.get(
            "text",
            "",
        )

        if not raw_text:
            continue

        lines = raw_text.splitlines()

        # ====================================================
        # PROCESS LINES
        # ====================================================

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Detect section heading.
            # ------------------------------------------------

            heading = find_section_heading(
                line
            )

            # ------------------------------------------------
            # NEW SECTION
            # ------------------------------------------------

            if heading:

                section_number, section_title = heading

                # Save previous section.
                if current_section is not None:

                    content = "\n".join(
                        current_section["content"]
                    )

                    current_section["content"] = (
                        clean_section_content(content)
                    )

                    sections.append(
                        current_section
                    )

                # Start new section.
                current_section = {
                    "section_number": section_number,
                    "section_title": section_title,
                    "page_number": page_number,
                    "content": [],
                }

                continue

            # ------------------------------------------------
            # SECTION CONTENT
            # ------------------------------------------------

            if current_section is not None:

                # Skip obvious standalone footnotes.
                if is_obvious_footnote(line):
                    continue

                current_section["content"].append(
                    line
                )

    # ========================================================
    # SAVE FINAL SECTION
    # ========================================================

    if current_section is not None:

        content = "\n".join(
            current_section["content"]
        )

        current_section["content"] = (
            clean_section_content(content)
        )

        sections.append(
            current_section
        )

    # ========================================================
    # REMOVE EMPTY / INVALID SECTIONS
    # ========================================================

    valid_sections = []

    for section in sections:

        number = section["section_number"]
        title = section["section_title"]
        content = section["content"]

        # A section must have a number and title.
        if not number or not title:
            continue

        # Keep section even if content is short.
        # Some sections are genuinely short.
        valid_sections.append(section)

    # ========================================================
    # REMOVE DUPLICATE SECTIONS
    # ========================================================

    # PDF documents can sometimes contain duplicate headings.
    # Keep the first legitimate occurrence.
    unique_sections = []

    seen_numbers = set()

    for section in valid_sections:

        number = section["section_number"]

        if number in seen_numbers:
            continue

        seen_numbers.add(number)

        unique_sections.append(section)

    return unique_sections


# ============================================================
# GET SECTION
# ============================================================

def get_section(
    sections: List[Dict],
    section_number: str,
) -> Optional[Dict]:
    """
    Retrieve a section by its section number.

    Example:

        section = get_section(sections, "73")
    """

    target = str(
        section_number
    ).strip().upper()

    for section in sections:

        current = str(
            section.get("section_number", "")
        ).strip().upper()

        if current == target:
            return section

    return None


# ============================================================
# VALIDATE SECTIONS
# ============================================================

def validate_sections(
    sections: List[Dict],
) -> Dict:
    """
    Validate parsed legal sections.

    Return keys are compatible with test_parser.py.
    """

    numbers = [
        str(section["section_number"])
        for section in sections
    ]

    # --------------------------------------------------------
    # Find duplicates.
    # --------------------------------------------------------

    duplicates = sorted(
        {
            number
            for number in numbers
            if numbers.count(number) > 1
        },
        key=lambda x: (
            int(re.match(r"\d+", x).group()),
            x,
        ),
    )

    # --------------------------------------------------------
    # Important sections used for validation.
    # --------------------------------------------------------

    important_sections = [
        "1",
        "2",
        "3",
        "10",
        "16",
        "19A",
        "20",
        "23",
        "28",
        "56",
        "73",
        "74",
        "124",
        "148",
        "151",
        "161",
        "178",
        "182",
    ]

    missing = [
        number
        for number in important_sections
        if number not in numbers
    ]

    return {
        "total_sections": len(sections),
        "first_section": (
            numbers[0]
            if numbers
            else None
        ),
        "last_section": (
            numbers[-1]
            if numbers
            else None
        ),

        # Both names are provided so older/newer
        # test scripts remain compatible.
        "duplicate_sections": duplicates,
        "duplicates": duplicates,

        "missing_sections": missing,
        "missing_required": missing,
    }