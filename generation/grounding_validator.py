"""
Legal RAG - Source-Aware Grounding Validator
--------------------------------------------

Validates generated answers against the exact retrieved
legal sources.

Checks:

1. Explicit legal section references
2. SOURCE N references
3. SOURCE N -> Section X consistency
4. Unsupported sections
5. Output format
6. Reasoning leakage
7. Thinking-tag leakage
8. Template placeholders
"""

import re


class GroundingValidator:

    def __init__(self):
        pass

    # ============================================================
    # RETRIEVED SOURCE MAP
    # ============================================================

    def build_source_map(self, results):
        """
        Build:

            SOURCE 1 -> Section 2
            SOURCE 2 -> Section 18
            SOURCE 3 -> Section 5

        This is based strictly on the retriever output order.
        """

        source_map = {}

        for index, result in enumerate(
            results,
            start=1
        ):

            section = str(
                result.get(
                    "section",
                    ""
                )
            ).strip()

            title = str(
                result.get(
                    "title",
                    ""
                )
            ).strip()

            source_map[str(index)] = {
                "section": section.upper(),
                "title": title
            }

        return source_map

    # ============================================================
    # RETRIEVED SECTIONS
    # ============================================================

    def get_retrieved_sections(self, results):

        sections = set()

        for result in results:

            section = str(
                result.get(
                    "section",
                    ""
                )
            ).strip()

            if section:

                sections.add(
                    section.upper()
                )

        return sections

    # ============================================================
    # EXPLICIT SECTION REFERENCES
    # ============================================================

    def extract_section_numbers(self, text):
        """
        Extract explicit legal section references.

        Examples:

            Section 2
            Section 35
            Sections 28, 42 and 53
            Section: 2
        """

        if not text:
            return set()

        sections = set()

        patterns = [

            r"\bSections?\s*:?\s*(\d+[A-Za-z]?)",

        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for match in matches:

                sections.add(
                    match.upper()
                )

        return sections

    # ============================================================
    # SOURCE REFERENCES
    # ============================================================

    def extract_source_numbers(self, text):
        """
        Extract references such as:

            SOURCE 1
            Source 2
            SOURCE 5

        Does NOT treat plain numbers as source numbers.
        """

        if not text:
            return set()

        matches = re.findall(
            r"\bSOURCE\s+(\d+)\b",
            text,
            flags=re.IGNORECASE
        )

        return {
            match
            for match in matches
        }

    # ============================================================
    # PLAIN SOURCE NUMBER LIST
    # ============================================================

    def extract_sources_field(self, answer):
        """
        Extract the contents of the Sources section.

        Example:

            Sources:
            1
            35
            81

        or:

            Sources:
            2, 18, 35
        """

        if not answer:
            return ""

        match = re.search(
            r"Sources:\s*(.*?)(?:\n\s*[A-Za-z ]+:|$)",
            answer,
            flags=re.IGNORECASE | re.DOTALL
        )

        if not match:
            return ""

        return match.group(1).strip()

    # ============================================================
    # SOURCE -> SECTION VALIDATION
    # ============================================================

    def validate_source_references(
        self,
        answer,
        source_map
    ):
        """
        Validate expressions such as:

            Section 7 of SOURCE 1

        against:

            SOURCE 1 -> Section 2
        """

        mismatches = []

        pattern = re.compile(
            r"\bSection\s*:?\s*(\d+[A-Za-z]?)"
            r".{0,100}?"
            r"\bSOURCE\s+(\d+)\b",
            flags=re.IGNORECASE | re.DOTALL
        )

        matches = pattern.findall(
            answer
        )

        for section, source_number in matches:

            section = section.upper()

            source_info = source_map.get(
                source_number
            )

            if source_info is None:

                mismatches.append(
                    {
                        "section": section,
                        "source": source_number,
                        "expected_section": None,
                        "reason": "Unknown source number"
                    }
                )

                continue

            expected_section = (
                source_info["section"]
            )

            if section != expected_section:

                mismatches.append(
                    {
                        "section": section,
                        "source": source_number,
                        "expected_section": expected_section,
                        "reason": "Section/source mismatch"
                    }
                )

        return {
            "valid": len(mismatches) == 0,
            "mismatches": mismatches
        }

    # ============================================================
    # SOURCES FIELD VALIDATION
    # ============================================================

    def validate_sources_field(
        self,
        answer,
        retrieved_sections
    ):
        """
        Validate the Sources: field.

        The Sources field should contain legal section numbers,
        not arbitrary source numbers.
        """

        sources_text = (
            self.extract_sources_field(
                answer
            )
        )

        if not sources_text:

            return {
                "valid": False,
                "reason": "Sources field is empty or missing",
                "unsupported_sections": []
            }

        # --------------------------------------------------------
        # Detect "Not applicable"
        # --------------------------------------------------------

        if re.search(
            r"\bnot\s+applicable\b",
            sources_text,
            flags=re.IGNORECASE
        ):

            return {
                "valid": True,
                "reason": None,
                "unsupported_sections": []
            }

        # --------------------------------------------------------
        # Extract section-like numbers
        # --------------------------------------------------------

        source_numbers = re.findall(
            r"\b\d+[A-Za-z]?\b",
            sources_text
        )

        cited_sections = {
            value.upper()
            for value in source_numbers
        }

        unsupported = (
            cited_sections
            - retrieved_sections
        )

        return {
            "valid": len(unsupported) == 0,
            "reason": (
                None
                if len(unsupported) == 0
                else "Sources contain sections not retrieved"
            ),
            "unsupported_sections": sorted(
                unsupported
            )
        }

    # ============================================================
    # SECTION GROUNDING
    # ============================================================

    def validate_section_grounding(
        self,
        answer,
        retrieved_sections
    ):
        """
        Check explicit legal section references.
        """

        cited_sections = (
            self.extract_section_numbers(
                answer
            )
        )

        unsupported = (
            cited_sections
            - retrieved_sections
        )

        return {
            "valid": len(unsupported) == 0,
            "cited_sections": sorted(
                cited_sections
            ),
            "retrieved_sections": sorted(
                retrieved_sections
            ),
            "unsupported_sections": sorted(
                unsupported
            )
        }

    # ============================================================
    # FORMAT VALIDATION
    # ============================================================

    def validate_format(self, answer):

        if not answer:

            return {
                "valid": False,
                "missing_sections": [
                    "Answer",
                    "Relevant Sections",
                    "Explanation",
                    "Sources"
                ]
            }

        required_headers = [
            "Answer:",
            "Relevant Sections:",
            "Explanation:",
            "Sources:"
        ]

        missing = []

        for header in required_headers:

            if header.lower() not in answer.lower():

                missing.append(
                    header
                )

        return {
            "valid": len(missing) == 0,
            "missing_sections": missing
        }

    # ============================================================
    # REASONING LEAKAGE
    # ============================================================

    def validate_reasoning(self, answer):

        patterns = [

            r"\bfirst,\s+i\b",
            r"\bfirst,\s+we\b",
            r"\blet's\s+analyze\b",
            r"\blet's\s+check\b",
            r"\blet's\s+look\b",
            r"\blet's\s+extract\b",
            r"\bi\s+need\s+to\s+answer\b",
            r"\bwe\s+need\s+to\b",
            r"\bthe\s+user\s+asks\b",
            r"\blooking\s+at\s+the\s+context\b",
            r"\bstep\s+\d+\s*:",
            r"\bi\s+should\b",
            r"\bwe\s+must\b",
        ]

        detected = []

        for pattern in patterns:

            if re.search(
                pattern,
                answer,
                flags=re.IGNORECASE
            ):

                detected.append(
                    pattern
                )

        return {
            "valid": len(detected) == 0,
            "detected_patterns": detected
        }

    # ============================================================
    # THINKING TAGS
    # ============================================================

    def validate_thinking_tags(self, answer):

        patterns = [
            r"<think>",
            r"</think>",
            r"<analysis>",
            r"</analysis>"
        ]

        detected = []

        for pattern in patterns:

            if re.search(
                pattern,
                answer,
                flags=re.IGNORECASE
            ):

                detected.append(
                    pattern
                )

        return {
            "valid": len(detected) == 0,
            "detected_tags": detected
        }

    # ============================================================
    # PLACEHOLDERS
    # ============================================================

    def validate_placeholders(self, answer):

        placeholders = [

            "[direct answer]",

            "[section number and title]",

            "[brief explanation based only on the supplied context]",

            "[section number(s)]"

        ]

        found = []

        answer_lower = answer.lower()

        for placeholder in placeholders:

            if placeholder.lower() in answer_lower:

                found.append(
                    placeholder
                )

        return {
            "valid": len(found) == 0,
            "placeholders": found
        }

    # ============================================================
    # FULL VALIDATION
    # ============================================================

    def validate(
        self,
        answer,
        results
    ):
        """
        Run every grounding check.
        """

        retrieved_sections = (
            self.get_retrieved_sections(
                results
            )
        )

        source_map = (
            self.build_source_map(
                results
            )
        )

        section_check = (
            self.validate_section_grounding(
                answer,
                retrieved_sections
            )
        )

        source_reference_check = (
            self.validate_source_references(
                answer,
                source_map
            )
        )

        sources_field_check = (
            self.validate_sources_field(
                answer,
                retrieved_sections
            )
        )

        format_check = (
            self.validate_format(
                answer
            )
        )

        reasoning_check = (
            self.validate_reasoning(
                answer
            )
        )

        thinking_check = (
            self.validate_thinking_tags(
                answer
            )
        )

        placeholder_check = (
            self.validate_placeholders(
                answer
            )
        )

        overall_valid = all(
            [
                section_check["valid"],
                source_reference_check["valid"],
                sources_field_check["valid"],
                format_check["valid"],
                reasoning_check["valid"],
                thinking_check["valid"],
                placeholder_check["valid"]
            ]
        )

        return {

            "valid": overall_valid,

            "source_map": source_map,

            "section_grounding": section_check,

            "source_references": source_reference_check,

            "sources_field": sources_field_check,

            "format": format_check,

            "reasoning": reasoning_check,

            "thinking_tags": thinking_check,

            "placeholders": placeholder_check
        }

    # ============================================================
    # DISPLAY
    # ============================================================

    def display_validation(
        self,
        validation
    ):

        print()
        print("=" * 80)
        print("GROUNDING VALIDATION")
        print("=" * 80)

        status = (
            "PASS"
            if validation["valid"]
            else "FAIL"
        )

        print(
            f"Overall: {status}"
        )

        # --------------------------------------------------------
        # Section grounding
        # --------------------------------------------------------

        section_check = (
            validation[
                "section_grounding"
            ]
        )

        print()
        print(
            "Section grounding: "
            + (
                "PASS"
                if section_check["valid"]
                else "FAIL"
            )
        )

        print(
            f"Cited sections: "
            f"{section_check['cited_sections']}"
        )

        print(
            f"Retrieved sections: "
            f"{section_check['retrieved_sections']}"
        )

        if section_check[
            "unsupported_sections"
        ]:

            print(
                f"Unsupported sections: "
                f"{section_check['unsupported_sections']}"
            )

        # --------------------------------------------------------
        # Source references
        # --------------------------------------------------------

        source_check = (
            validation[
                "source_references"
            ]
        )

        print()
        print(
            "Source references: "
            + (
                "PASS"
                if source_check["valid"]
                else "FAIL"
            )
        )

        if source_check["mismatches"]:

            for mismatch in (
                source_check["mismatches"]
            ):

                print(
                    f"  Section "
                    f"{mismatch['section']} "
                    f"claimed from SOURCE "
                    f"{mismatch['source']}, "
                    f"but SOURCE "
                    f"{mismatch['source']} "
                    f"is Section "
                    f"{mismatch['expected_section']}"
                )

        # --------------------------------------------------------
        # Sources field
        # --------------------------------------------------------

        sources_check = (
            validation[
                "sources_field"
            ]
        )

        print()
        print(
            "Sources field: "
            + (
                "PASS"
                if sources_check["valid"]
                else "FAIL"
            )
        )

        if sources_check[
            "unsupported_sections"
        ]:

            print(
                "Unsupported source sections: "
                f"{sources_check['unsupported_sections']}"
            )

        # --------------------------------------------------------
        # Format
        # --------------------------------------------------------

        format_check = (
            validation[
                "format"
            ]
        )

        print()
        print(
            "Output format: "
            + (
                "PASS"
                if format_check["valid"]
                else "FAIL"
            )
        )

        if format_check[
            "missing_sections"
        ]:

            print(
                "Missing headers: "
                f"{format_check['missing_sections']}"
            )

        # --------------------------------------------------------
        # Reasoning
        # --------------------------------------------------------

        reasoning_check = (
            validation[
                "reasoning"
            ]
        )

        print(
            "Reasoning leakage: "
            + (
                "PASS"
                if reasoning_check["valid"]
                else "FAIL"
            )
        )

        # --------------------------------------------------------
        # Thinking
        # --------------------------------------------------------

        thinking_check = (
            validation[
                "thinking_tags"
            ]
        )

        print(
            "Thinking tags: "
            + (
                "PASS"
                if thinking_check["valid"]
                else "FAIL"
            )
        )

        # --------------------------------------------------------
        # Placeholders
        # --------------------------------------------------------

        placeholder_check = (
            validation[
                "placeholders"
            ]
        )

        print(
            "Template placeholders: "
            + (
                "PASS"
                if placeholder_check["valid"]
                else "FAIL"
            )
        )