"""
Claim-Level Grounding Validator
================================

Validates individual claims in a generated legal answer against
the retrieved legal context.

This validator is intentionally conservative:

1. Splits the answer into meaningful claims.
2. Measures semantic similarity between each claim and retrieved context.
3. Measures lexical overlap.
4. Detects a small number of targeted legal contradictions.
5. Classifies each claim as:
      - SUPPORTED
      - PARTIALLY_SUPPORTED
      - UNSUPPORTED
6. Produces an overall grounding status.

Important:
This is a grounding signal, NOT a legal truth detector.
Semantic similarity alone is never treated as proof of legal correctness.
"""

import re
from typing import List, Dict, Any, Optional


class ClaimGroundingValidator:
    """
    Validate individual claims against retrieved legal sections.
    """

    def __init__(
        self,
        embedding_model=None,
        semantic_supported_threshold: float = 0.58,
        semantic_partial_threshold: float = 0.42,
        lexical_supported_threshold: float = 0.25,
    ):
        """
        Parameters
        ----------
        embedding_model:
            Existing SentenceTransformer model from the retriever.
            Passing the existing model avoids loading another copy.

        semantic_supported_threshold:
            Minimum semantic similarity for strong support.

        semantic_partial_threshold:
            Minimum semantic similarity for partial support.

        lexical_supported_threshold:
            Minimum lexical overlap used together with semantic similarity.
        """

        self.model = embedding_model

        self.semantic_supported_threshold = semantic_supported_threshold
        self.semantic_partial_threshold = semantic_partial_threshold
        self.lexical_supported_threshold = lexical_supported_threshold

    # ============================================================
    # TEXT CLEANING
    # ============================================================

    def _clean_answer(self, answer: str) -> str:
        """
        Clean generated answer before claim extraction.
        """

        if not answer:
            return ""

        text = str(answer)

        # Remove <think>...</think>
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove markdown code fences
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # Normalize Windows line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive spaces
        text = re.sub(r"[ \t]+", " ", text)

        # IMPORTANT:
        # Join lines because generated answers can wrap one sentence
        # across several physical lines.
        lines = []

        for line in text.split("\n"):
            line = line.strip()

            if not line:
                continue

            lines.append(line)

        text = " ".join(lines)

        # Remove excessive whitespace again
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _remove_headers(self, text: str) -> str:
        """
        Remove common answer-format headers so they do not become claims.
        """

        headers = [
            r"\bAnswer\s*:",
            r"\bRelevant Sections\s*:",
            r"\bExplanation\s*:",
            r"\bSources\s*:",
            r"\bSource\s*:",
        ]

        for pattern in headers:
            text = re.sub(
                pattern,
                " ",
                text,
                flags=re.IGNORECASE,
            )

        return re.sub(r"\s+", " ", text).strip()

    # ============================================================
    # CLAIM EXTRACTION
    # ============================================================

    def extract_claims(self, answer: str) -> List[str]:
        """
        Extract meaningful claims from a generated answer.

        Handles:
        - normal sentences
        - numbered lists
        - bullet points
        - line-wrapped sentences
        """

        text = self._clean_answer(answer)

        if not text:
            return []

        text = self._remove_headers(text)

        # Remove common list markers
        text = re.sub(
            r"(?m)(^|\s)[•*-]\s+",
            " ",
            text,
        )

        # Remove numbered list markers such as:
        # 1. ...
        # 2) ...
        text = re.sub(
            r"(?m)(^|\s)\d+[\.\)]\s+",
            " ",
            text,
        )

        # Split on sentence-ending punctuation.
        #
        # We intentionally avoid splitting on every colon/semicolon
        # because legal answers can contain section references and
        # structured phrases.
        raw_claims = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        claims = []

        for claim in raw_claims:
            claim = claim.strip()

            if not claim:
                continue

            # Remove accidental leading punctuation
            claim = re.sub(
                r"^[\s,;:]+",
                "",
                claim,
            ).strip()

            if not claim:
                continue

            # Ignore very short fragments
            words = claim.split()

            if len(words) < 4:
                continue

            # Ignore obvious source-only fragments
            if re.fullmatch(
                r"(source|sources?)\s*:?\s*[\d,\sandsection-]+",
                claim,
                flags=re.IGNORECASE,
            ):
                continue

            claims.append(claim)

        return claims

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for lexical comparison.
        """

        if not text:
            return ""

        text = text.lower()

        # Normalize section references
        text = re.sub(
            r"section\s+(\d+[a-z]?)",
            r"section \1",
            text,
        )

        # Remove punctuation
        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        # Normalize whitespace
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def _get_words(self, text: str) -> set:
        """
        Return meaningful words for lexical comparison.
        """

        normalized = self.normalize_text(text)

        words = normalized.split()

        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "of",
            "to",
            "for",
            "and",
            "or",
            "in",
            "on",
            "at",
            "by",
            "with",
            "under",
            "as",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "from",
            "who",
            "which",
            "where",
            "when",
            "how",
            "can",
            "may",
            "shall",
            "such",
        }

        return {
            word
            for word in words
            if word not in stop_words and len(word) > 2
        }

    # ============================================================
    # LEXICAL SUPPORT
    # ============================================================

    def lexical_support(
        self,
        claim: str,
        source_text: str,
    ) -> float:
        """
        Calculate lexical overlap between claim and source.

        Returns:
            float between 0.0 and 1.0
        """

        claim_words = self._get_words(claim)
        source_words = self._get_words(source_text)

        if not claim_words:
            return 0.0

        overlap = claim_words.intersection(source_words)

        return len(overlap) / len(claim_words)

    # ============================================================
    # LEGAL CONTRADICTION DETECTION
    # ============================================================

    def _has_exclusion_language(self, source_text: str) -> bool:
        """
        Detect whether the source contains explicit exclusion language.
        """

        source_lower = self.normalize_text(source_text)

        exclusion_phrases = [
            "does not include",
            "does not apply",
            "not include",
            "not apply",
            "excluding",
            "excluded",
            "shall not include",
            "shall not apply",
            "other than",
            "except",
            "unless",
        ]

        return any(
            phrase in source_lower
            for phrase in exclusion_phrases
        )

    def _detect_legal_contradiction(
        self,
        claim: str,
        source_text: str,
    ) -> bool:
        """
        Detect targeted legal contradictions.

        IMPORTANT:
        This deliberately avoids generic negation detection.

        Example of a contradiction:

            Claim:
            "A consumer is a person who buys goods exclusively
             for resale and commercial purposes."

            Source:
            "does not include a person who obtains such goods
             for resale or for any commercial purpose."

        Semantic similarity may be high because both texts discuss
        consumers and commercial purchases. The legal polarity,
        however, is opposite.

        Returns:
            True  -> likely legal contradiction
            False -> no targeted contradiction detected
        """

        claim_lower = self.normalize_text(claim)
        source_lower = self.normalize_text(source_text)

        # --------------------------------------------------------
        # CONSUMER DEFINITION:
        # RESALE EXCLUSION
        # --------------------------------------------------------

        claim_mentions_resale = any(
            phrase in claim_lower
            for phrase in [
                "for resale",
                "resale",
                "reselling",
                "for commercial purpose",
                "for commercial purposes",
                "commercial purpose",
                "commercial purposes",
            ]
        )

        if claim_mentions_resale:

            source_has_resale_exclusion = (
                "does not include a person who obtains such goods for resale"
                in source_lower
                or "obtains such goods for resale"
                in source_lower
                or "for resale or for any commercial purpose"
                in source_lower
                or (
                    "for resale" in source_lower
                    and "does not include" in source_lower
                )
            )

            if source_has_resale_exclusion:

                # The claim presents the person as a consumer.
                consumer_inclusion_patterns = [
                    "a consumer is a person",
                    "a consumer is someone",
                    "is a consumer",
                    "is considered a consumer",
                    "qualifies as a consumer",
                    "qualify as a consumer",
                    "included as a consumer",
                    "included in the definition of consumer",
                    "comes within the definition of consumer",
                    "falls within the definition of consumer",
                ]

                claim_presents_as_consumer = any(
                    pattern in claim_lower
                    for pattern in consumer_inclusion_patterns
                )

                if claim_presents_as_consumer:
                    return True

        # --------------------------------------------------------
        # CONSUMER DEFINITION:
        # COMMERCIAL PURPOSE EXCLUSION
        # --------------------------------------------------------

        claim_mentions_commercial = any(
            phrase in claim_lower
            for phrase in [
                "commercial purpose",
                "commercial purposes",
            ]
        )

        if claim_mentions_commercial:

            source_has_commercial_exclusion = (
                "commercial purpose" in source_lower
                and (
                    "does not include" in source_lower
                    or "excluding" in source_lower
                    or "shall not include" in source_lower
                )
            )

            if source_has_commercial_exclusion:

                positive_commercial_claim_patterns = [
                    "commercial purpose qualifies",
                    "commercial purposes qualify",
                    "commercial purpose is a consumer",
                    "commercial purposes is a consumer",
                    "commercial purpose makes a person a consumer",
                    "commercial purposes make a person a consumer",
                    "buying for commercial purpose is a consumer",
                    "buying for commercial purposes is a consumer",
                    "a person buying for commercial purpose is a consumer",
                    "a person buying for commercial purposes is a consumer",
                ]

                if any(
                    pattern in claim_lower
                    for pattern in positive_commercial_claim_patterns
                ):
                    return True

                # Broad construction:
                # If claim explicitly connects "commercial purpose"
                # with being a consumer, flag it.
                if (
                    "commercial purpose" in claim_lower
                    and (
                        "is a consumer" in claim_lower
                        or "are consumers" in claim_lower
                        or "qualifies as a consumer" in claim_lower
                        or "qualify as a consumer" in claim_lower
                    )
                ):
                    return True

        return False

    # ============================================================
    # SEMANTIC SUPPORT
    # ============================================================

    def semantic_support(
        self,
        claim: str,
        source_text: str,
    ) -> float:
        """
        Calculate cosine similarity between claim and source.

        Uses the embedding model already loaded by the retriever.
        """

        if self.model is None:
            return 0.0

        if not claim or not source_text:
            return 0.0

        try:
            embeddings = self.model.encode(
                [claim, source_text],
                normalize_embeddings=True,
            )

            similarity = float(
                embeddings[0] @ embeddings[1]
            )

            # Guard against tiny floating point errors
            similarity = max(
                0.0,
                min(1.0, similarity),
            )

            return similarity

        except Exception:
            return 0.0

    # ============================================================
    # CLAIM CLASSIFICATION
    # ============================================================

    def classify_support(
        self,
        claim: str,
        source_text: str,
    ) -> Dict[str, Any]:
        """
        Classify a single claim.

        Returns a dictionary containing:
        - status
        - semantic_similarity
        - lexical_support
        - legal_contradiction
        """

        semantic_score = self.semantic_support(
            claim,
            source_text,
        )

        lexical_score = self.lexical_support(
            claim,
            source_text,
        )

        contradiction = self._detect_legal_contradiction(
            claim,
            source_text,
        )

        # --------------------------------------------------------
        # CONTRADICTION OVERRIDE
        # --------------------------------------------------------
        #
        # A direct legal contradiction must override semantic
        # similarity. Otherwise a false statement about an excluded
        # category can appear "supported" because it uses similar
        # terminology to the source.
        # --------------------------------------------------------

        if contradiction:

            return {
                "status": "UNSUPPORTED",
                "semantic_similarity": semantic_score,
                "lexical_support": lexical_score,
                "legal_contradiction": True,
            }

        # --------------------------------------------------------
        # STRONG SUPPORT
        # --------------------------------------------------------

        if (
            semantic_score >= self.semantic_supported_threshold
            and lexical_score >= self.lexical_supported_threshold
        ):
            status = "SUPPORTED"

        # --------------------------------------------------------
        # ALSO ALLOW VERY HIGH SEMANTIC SUPPORT
        # --------------------------------------------------------

        elif semantic_score >= 0.70:
            status = "SUPPORTED"

        # --------------------------------------------------------
        # PARTIAL SUPPORT
        # --------------------------------------------------------

        elif semantic_score >= self.semantic_partial_threshold:

            # If lexical overlap is reasonable, classify as partial.
            if lexical_score >= 0.10:
                status = "PARTIALLY_SUPPORTED"

            else:
                status = "PARTIALLY_SUPPORTED"

        # --------------------------------------------------------
        # UNSUPPORTED
        # --------------------------------------------------------

        else:
            status = "UNSUPPORTED"

        return {
            "status": status,
            "semantic_similarity": semantic_score,
            "lexical_support": lexical_score,
            "legal_contradiction": False,
        }

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate(
        self,
        answer: str,
        retrieved_sections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate every claim in an answer against retrieved sections.

        Parameters
        ----------
        answer:
            Generated legal answer.

        retrieved_sections:
            List of retriever metadata dictionaries.

        Returns
        -------
        dict
            Complete validation result.
        """

        claims = self.extract_claims(answer)

        results = []

        # --------------------------------------------------------
        # Prepare source text
        # --------------------------------------------------------

        for claim in claims:

            best_result: Optional[Dict[str, Any]] = None

            for section in retrieved_sections:

                source_text = section.get(
                    "content",
                    section.get(
                        "text",
                        "",
                    ),
                )

                if not source_text:
                    continue

                classification = self.classify_support(
                    claim,
                    source_text,
                )

                result = {
                    "claim": claim,
                    "status": classification["status"],
                    "semantic_similarity": classification[
                        "semantic_similarity"
                    ],
                    "lexical_support": classification[
                        "lexical_support"
                    ],
                    "legal_contradiction": classification[
                        "legal_contradiction"
                    ],
                    "supporting_section": section.get(
                        "section_number"
                    ),
                    "supporting_title": section.get(
                        "section_title",
                        section.get("title", ""),
                    ),
                }

                # ------------------------------------------------
                # Select best source.
                #
                # Contradictions are prioritized over ordinary
                # similarity because they are legally important.
                # ------------------------------------------------

                if best_result is None:
                    best_result = result

                elif (
                    result["legal_contradiction"]
                    and not best_result["legal_contradiction"]
                ):
                    best_result = result

                else:
                    current_score = (
                        0.65
                        * result["semantic_similarity"]
                        + 0.35
                        * result["lexical_support"]
                    )

                    best_score = (
                        0.65
                        * best_result["semantic_similarity"]
                        + 0.35
                        * best_result["lexical_support"]
                    )

                    if current_score > best_score:
                        best_result = result

            if best_result is not None:
                results.append(best_result)

        # --------------------------------------------------------
        # Count statuses
        # --------------------------------------------------------

        total_claims = len(results)

        supported_count = sum(
            1
            for result in results
            if result["status"] == "SUPPORTED"
        )

        partial_count = sum(
            1
            for result in results
            if result["status"] == "PARTIALLY_SUPPORTED"
        )

        unsupported_count = sum(
            1
            for result in results
            if result["status"] == "UNSUPPORTED"
        )

        # --------------------------------------------------------
        # Overall status
        # --------------------------------------------------------

        if total_claims == 0:

            overall_status = "NO_CLAIMS"

        elif unsupported_count > 0:

            overall_status = "UNSUPPORTED_CLAIMS_FOUND"

        elif partial_count > 0:

            overall_status = "PARTIAL_SUPPORT"

        else:

            overall_status = "FULLY_SUPPORTED"

        return {
            "overall_status": overall_status,
            "total_claims": total_claims,
            "supported": supported_count,
            "partially_supported": partial_count,
            "unsupported": unsupported_count,
            "claims": results,
        }

    # ============================================================
    # DISPLAY
    # ============================================================

    def display_validation(
        self,
        validation_result: Dict[str, Any],
    ) -> None:
        """
        Pretty-print validation results.
        """

        print()
        print("=" * 80)
        print("CLAIM-LEVEL GROUNDING VALIDATION")
        print("=" * 80)

        print()
        print(
            f"Overall status: "
            f"{validation_result['overall_status']}"
        )

        print(
            f"Total claims: "
            f"{validation_result['total_claims']}"
        )

        print(
            f"Supported: "
            f"{validation_result['supported']}"
        )

        print(
            f"Partially supported: "
            f"{validation_result['partially_supported']}"
        )

        print(
            f"Unsupported: "
            f"{validation_result['unsupported']}"
        )

        print()
        print("-" * 80)

        for index, claim_result in enumerate(
            validation_result["claims"],
            start=1,
        ):

            print(
                f"Claim {index}:"
            )

            print(
                claim_result["claim"]
            )

            print()

            print(
                f"Status: "
                f"{claim_result['status']}"
            )

            print(
                f"Semantic similarity: "
                f"{claim_result['semantic_similarity']:.4f}"
            )

            print(
                f"Lexical support: "
                f"{claim_result['lexical_support']:.4f}"
            )

            print(
                f"Legal contradiction: "
                f"{claim_result['legal_contradiction']}"
            )

            print(
                f"Supporting section: "
                f"{claim_result['supporting_section']}"
            )

            if claim_result.get("supporting_title"):
                print(
                    f"Supporting title: "
                    f"{claim_result['supporting_title']}"
                )

            print()
            print("-" * 80)


# ================================================================
# FACTORY FUNCTION
# ================================================================

def create_claim_grounding_validator(
    embedding_model=None,
) -> ClaimGroundingValidator:
    """
    Factory function.

    Pass the embedding model already loaded by the retriever
    so we don't load another SentenceTransformer model.
    """

    return ClaimGroundingValidator(
        embedding_model=embedding_model,
        semantic_supported_threshold=0.58,
        semantic_partial_threshold=0.42,
        lexical_supported_threshold=0.25,
    )