# retrieval/retriever.py

import os
import re
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# Memory-safe environment settings
# ============================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 15


# ============================================================
# Legal Retriever
# ============================================================

class LegalRetriever:

    def __init__(
        self,
        index_path,
        metadata_path,
        model_name=DEFAULT_MODEL
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name

        print("Loading FAISS index...")
        self.index = faiss.read_index(index_path)

        print("Loading metadata...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print(f"Index vectors : {self.index.ntotal}")
        print(f"Metadata items: {len(self.metadata)}")
        print(f"Embedding dim : {self.index.d}")

    # ========================================================
    # Metadata helpers
    # ========================================================

    def _get_content(self, item):
        """
        Get chunk text from metadata.

        Current metadata format:
        {
            "chunk_id": ...,
            "section_number": ...,
            "section_title": ...,
            "chapter": ...,
            "chunk_number": ...,
            "total_chunks": ...,
            "content": ...
        }
        """

        return str(
            item.get(
                "content",
                item.get("text", "")
            )
        )

    def _get_title(self, item):
        return str(
            item.get(
                "section_title",
                item.get("title", "")
            )
        )

    def _get_section_number(self, item):
        return str(
            item.get(
                "section_number",
                item.get("section", "")
            )
        )

    # ========================================================
    # Text normalization
    # ========================================================

    def _normalize_text(self, text):
        text = str(text).lower()

        # Normalize section notation
        text = re.sub(r"section\s+", "section ", text)

        # Replace punctuation with spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ========================================================
    # Keyword extraction
    # ========================================================

    def _extract_keywords(self, query):

        text = self._normalize_text(query)

        stopwords = {
            "what",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "under",
            "this",
            "act",
            "can",
            "how",
            "does",
            "do",
            "to",
            "for",
            "in",
            "on",
            "and",
            "or",
            "be",
            "by",
            "with",
            "from",
            "which",
            "different",
            "about",
            "explain",
            "tell",
            "me"
        }

        words = text.split()

        keywords = [
            word
            for word in words
            if word not in stopwords and len(word) > 2
        ]

        return keywords

    # ========================================================
    # Query detectors
    # ========================================================

    def _is_definition_query(self, query):

        q = self._normalize_text(query)

        definition_patterns = [
            "what is",
            "what are",
            "meaning of",
            "define",
            "definition of",
            "who is",
            "who are"
        ]

        return any(
            pattern in q
            for pattern in definition_patterns
        )

    # --------------------------------------------------------

    def _is_commission_structure_query(self, query):

        q = self._normalize_text(query)

        patterns = [
            "consumer commission",
            "consumer commissions",
            "different commissions",
            "types of commissions",
            "commission structure",
            "district commission",
            "state commission",
            "national commission",
            "three tier",
            "three tier commission",
            "three tier system",
            "levels of commission",
            "consumer dispute redressal commission"
        ]

        return any(
            pattern in q
            for pattern in patterns
        )

    # --------------------------------------------------------

    def _is_complaint_filing_query(self, query):

        q = self._normalize_text(query)

        patterns = [
            "file a complaint",
            "file complaint",
            "filing a complaint",
            "filing complaint",
            "how can a consumer complain",
            "how can consumer complain",
            "where can a consumer file",
            "where can consumer file",
            "manner in which complaint",
            "make a complaint",
            "submit a complaint",
            "submit complaint"
        ]

        return any(
            pattern in q
            for pattern in patterns
        )

    # ========================================================
    # Structural scoring
    # ========================================================

    def _structural_score(self, query, section_number):

        section_number = str(section_number)

        # ----------------------------------------------------
        # Consumer commission structure
        # ----------------------------------------------------

        if self._is_commission_structure_query(query):

            commission_scores = {
                "28": 1.00,   # District Commission
                "42": 1.00,   # State Commission
                "53": 1.00    # National Commission
            }

            return commission_scores.get(
                section_number,
                0.0
            )

        # ----------------------------------------------------
        # Complaint filing
        # ----------------------------------------------------

        if self._is_complaint_filing_query(query):

            complaint_scores = {
                "35": 1.00,   # Manner in which complaint shall be made
                "38": 0.85,   # Procedure on admission of complaint
                "17": 0.45    # Complaints to authorities
            }

            return complaint_scores.get(
                section_number,
                0.0
            )

        return 0.0

    # ========================================================
    # Lexical scoring
    # ========================================================

    def _lexical_score(self, query, item):

        content = self._normalize_text(
            self._get_content(item)
        )

        title = self._normalize_text(
            self._get_title(item)
        )

        query_normalized = self._normalize_text(query)

        keywords = self._extract_keywords(query)

        if not keywords:
            return 0.0

        # ----------------------------------------------------
        # Keyword coverage
        # ----------------------------------------------------

        keyword_hits = 0

        for keyword in keywords:

            if keyword in content:
                keyword_hits += 1

        keyword_coverage = (
            keyword_hits / len(keywords)
        )

        # ----------------------------------------------------
        # Title relevance
        # ----------------------------------------------------

        title_hits = 0

        for keyword in keywords:

            if keyword in title:
                title_hits += 1

        title_score = (
            title_hits / len(keywords)
        )

        # ----------------------------------------------------
        # Exact phrase match
        # ----------------------------------------------------

        exact_phrase_score = 0.0

        if query_normalized in content:

            exact_phrase_score = 1.0

        # ----------------------------------------------------
        # Special legal concepts
        # ----------------------------------------------------

        concept_score = 0.0

        # Consumer definition
        if "consumer" in query_normalized:

            if "consumer" in title:
                concept_score += 0.30

            if "definition" in query_normalized:
                if "consumer" in title:
                    concept_score += 0.30

        # Consumer rights
        if (
            "consumer rights" in query_normalized
            or "rights of consumers" in query_normalized
        ):

            if "consumer rights" in title:
                concept_score += 0.60

            if "rights" in content:
                concept_score += 0.20

        # Product liability
        if "product liability" in query_normalized:

            if "product liability" in title:
                concept_score += 0.70

            if "product liability" in content:
                concept_score += 0.20

        # Commission structure
        if self._is_commission_structure_query(query):

            if "commission" in title:
                concept_score += 0.50

            if (
                "district" in content
                or "state" in content
                or "national" in content
            ):
                concept_score += 0.20

        # Complaint filing
        if self._is_complaint_filing_query(query):

            if "complaint" in title:
                concept_score += 0.50

            if "complaint" in content:
                concept_score += 0.20

        # ----------------------------------------------------
        # Final lexical score
        # ----------------------------------------------------

        score = (
            0.45 * keyword_coverage
            + 0.25 * title_score
            + 0.15 * exact_phrase_score
            + 0.15 * concept_score
        )

        return min(score, 1.0)

    # ========================================================
    # Combined score
    # ========================================================

    def _combined_score(
        self,
        semantic_score,
        lexical_score,
        structural_score
    ):

        return (
            0.55 * semantic_score
            + 0.20 * lexical_score
            + 0.25 * structural_score
        )

    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        candidate_k=DEFAULT_CANDIDATE_K
    ):

        # ----------------------------------------------------
        # Encode query
        # ----------------------------------------------------

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Semantic search
        # ----------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            candidate_k
        )

        candidates = []

        # ----------------------------------------------------
        # Build candidate results
        # ----------------------------------------------------

        for rank, index in enumerate(indices[0]):

            if index < 0:
                continue

            if index >= len(self.metadata):
                continue

            item = self.metadata[index]

            semantic_score = float(
                scores[0][rank]
            )

            lexical_score = self._lexical_score(
                query,
                item
            )

            section_number = self._get_section_number(
                item
            )

            structural_score = self._structural_score(
                query,
                section_number
            )

            combined_score = self._combined_score(
                semantic_score,
                lexical_score,
                structural_score
            )

            candidates.append(
                {
                    "rank": rank + 1,
                    "index": int(index),
                    "section_number": section_number,
                    "section_title": self._get_title(item),
                    "chapter": item.get(
                        "chapter",
                        ""
                    ),
                    "chunk_id": item.get(
                        "chunk_id",
                        ""
                    ),
                    "chunk_number": item.get(
                        "chunk_number",
                        1
                    ),
                    "content": self._get_content(item),
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "structural_score": structural_score,
                    "combined_score": combined_score
                }
            )

        # ====================================================
        # Force required structural sections into candidates
        # ====================================================

        required_sections = []

        if self._is_commission_structure_query(query):

            required_sections = [
                "28",
                "42",
                "53"
            ]

        elif self._is_complaint_filing_query(query):

            required_sections = [
                "35",
                "38"
            ]

        # ----------------------------------------------------
        # Add required sections if absent
        # ----------------------------------------------------

        existing_sections = {
            result["section_number"]
            for result in candidates
        }

        for required_section in required_sections:

            if required_section in existing_sections:
                continue

            # Search metadata for required section
            for index, item in enumerate(self.metadata):

                section_number = self._get_section_number(
                    item
                )

                if section_number != required_section:
                    continue

                lexical_score = self._lexical_score(
                    query,
                    item
                )

                structural_score = self._structural_score(
                    query,
                    section_number
                )

                # --------------------------------------------
                # Important:
                #
                # These sections were injected because
                # structural rules identify them as mandatory.
                #
                # Give them a reasonable effective semantic
                # score so they do not disappear during ranking.
                # --------------------------------------------

                effective_semantic_score = 0.70

                combined_score = self._combined_score(
                    effective_semantic_score,
                    lexical_score,
                    structural_score
                )

                candidates.append(
                    {
                        "rank": 999,
                        "index": int(index),
                        "section_number": section_number,
                        "section_title": self._get_title(item),
                        "chapter": item.get(
                            "chapter",
                            ""
                        ),
                        "chunk_id": item.get(
                            "chunk_id",
                            ""
                        ),
                        "chunk_number": item.get(
                            "chunk_number",
                            1
                        ),
                        "content": self._get_content(item),
                        "semantic_score": effective_semantic_score,
                        "lexical_score": lexical_score,
                        "structural_score": structural_score,
                        "combined_score": combined_score
                    }
                )

                break

        # ====================================================
        # Sort by combined score
        # ====================================================

        candidates.sort(
            key=lambda x: x["combined_score"],
            reverse=True
        )

        # ====================================================
        # Preserve mandatory structural sections
        # ====================================================

        if required_sections:

            required = []
            others = []

            for result in candidates:

                if result["section_number"] in required_sections:

                    required.append(result)

                else:

                    others.append(result)

            # Keep required sections in logical Act order
            required.sort(
                key=lambda x:
                required_sections.index(
                    x["section_number"]
                )
            )

            # Remove accidental duplicates
            seen_required = set()
            unique_required = []

            for result in required:

                section = result["section_number"]

                if section in seen_required:
                    continue

                seen_required.add(section)
                unique_required.append(result)

            required = unique_required

            remaining_slots = max(
                0,
                top_k - len(required)
            )

            candidates = (
                required
                + others[:remaining_slots]
            )

        else:

            candidates = candidates[:top_k]

        # ====================================================
        # Re-number final results
        # ====================================================

        for i, result in enumerate(candidates, start=1):

            result["final_rank"] = i

        return candidates

    # ========================================================
    # Search and display
    # ========================================================

    def search_and_display(
        self,
        query,
        top_k=DEFAULT_TOP_K
    ):

        results = self.search(
            query,
            top_k=top_k
        )

        print()
        print("=" * 80)
        print("QUERY")
        print("=" * 80)

        print(query)

        print()
        print("=" * 80)
        print("RETRIEVED RESULTS")
        print("=" * 80)

        for result in results:

            print()
            print(
                f"Rank {result['final_rank']}"
            )

            print(
                f"Section: "
                f"{result['section_number']}"
            )

            print(
                f"Title: "
                f"{result['section_title']}"
            )

            print(
                f"Semantic Score: "
                f"{result['semantic_score']:.4f}"
            )

            print(
                f"Lexical Score: "
                f"{result['lexical_score']:.4f}"
            )

            print(
                f"Structural Score: "
                f"{result['structural_score']:.4f}"
            )

            print(
                f"Combined Score: "
                f"{result['combined_score']:.4f}"
            )

            print("-" * 80)

            content = result["content"]

            if len(content) > 500:
                content = content[:500] + "..."

            print(content)

        return results


# ============================================================
# Factory
# ============================================================

def create_retriever(
    index_path="vectorstore/cpa_2019.index",
    metadata_path="vectorstore/cpa_2019_metadata.json"
):

    return LegalRetriever(
        index_path=index_path,
        metadata_path=metadata_path
    )


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    INDEX_PATH = os.path.join(
        "vectorstore",
        "cpa_2019.index"
    )

    METADATA_PATH = os.path.join(
        "vectorstore",
        "cpa_2019_metadata.json"
    )

    retriever = LegalRetriever(
        INDEX_PATH,
        METADATA_PATH
    )

    query = input(
        "\nEnter your legal question: "
    )

    retriever.search_and_display(
        query,
        top_k=5
    )