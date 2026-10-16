"""
LEGAL RAG SYSTEM
Consumer Protection Act, 2019

Step 17B:
End-to-End Legal RAG Application with Grounding Validation

Pipeline:
    User Question
        ↓
    FAISS Retrieval
        ↓
    Context Builder
        ↓
    Qwen2.5 3B via Ollama
        ↓
    Grounding Validator
        ↓
    Grounded Legal Answer
        ↓
    Retrieved Legal Sources
"""

import os
import sys


# ================================================================
# MEMORY-SAFE SETTINGS
# ================================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ================================================================
# IMPORT PROJECT COMPONENTS
# ================================================================

from retrieval.retriever import LegalRetriever
from generation.answer_generator import LegalAnswerGenerator
from generation.grounding_validator import GroundingValidator


# ================================================================
# IMPORT CONTEXT BUILDER
# ================================================================

LegalContextBuilder = None
ContextBuilder = None

try:
    from retrieval.context_builder import LegalContextBuilder
except ImportError:
    try:
        from retrieval.context_builder import ContextBuilder
    except ImportError:
        pass


# ================================================================
# CONFIGURATION
# ================================================================

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

OLLAMA_MODEL = "qwen2.5:3b"

OLLAMA_URL = "http://localhost:11434"

TOP_K = 5


# ================================================================
# DISPLAY HELPERS
# ================================================================

def print_separator():
    print("-" * 70)


def print_header():
    """Display application header."""

    print()
    print("=" * 70)
    print("        LEGAL RAG - CONSUMER PROTECTION ACT, 2019")
    print("=" * 70)
    print()
    print("Local Legal Question Answering System")
    print("Knowledge Base: Consumer Protection Act, 2019")
    print("LLM: Qwen2.5 3B via Ollama")
    print("Grounding Validation: Enabled")
    print()


# ================================================================
# CLEAN ANSWER FOR DISPLAY
# ================================================================

def clean_answer_for_display(answer):
    """
    Clean formatting problems that may occasionally be produced
    by the local LLM.

    This function does NOT change legal content.
    """

    if not answer:
        return ""

    lines = answer.splitlines()

    cleaned_lines = []

    previous_header = None

    for line in lines:

        stripped = line.strip()

        # --------------------------------------------------------
        # Remove duplicate consecutive headers
        # --------------------------------------------------------

        if stripped in {
            "Answer:",
            "Relevant Sections:",
            "Explanation:",
            "Sources:"
        }:

            if stripped == previous_header:
                continue

            previous_header = stripped

        else:

            previous_header = None

        cleaned_lines.append(line)

    answer = "\n".join(
        cleaned_lines
    ).strip()

    # ------------------------------------------------------------
    # Remove repeated blank lines
    # ------------------------------------------------------------

    while "\n\n\n" in answer:

        answer = answer.replace(
            "\n\n\n",
            "\n\n"
        )

    return answer.strip()


# ================================================================
# CONTEXT BUILDER
# ================================================================

def build_context(results):
    """
    Build legal context from retrieved chunks.

    Uses the project's existing context builder when available.
    Otherwise creates a simple fallback context.
    """

    # ------------------------------------------------------------
    # Existing LegalContextBuilder
    # ------------------------------------------------------------

    if LegalContextBuilder is not None:

        try:

            builder = LegalContextBuilder()

            if hasattr(
                builder,
                "build_context"
            ):
                return builder.build_context(
                    results
                )

            if hasattr(
                builder,
                "create_context"
            ):
                return builder.create_context(
                    results
                )

            if hasattr(
                builder,
                "build"
            ):
                return builder.build(
                    results
                )

        except Exception:
            pass

    # ------------------------------------------------------------
    # Existing ContextBuilder
    # ------------------------------------------------------------

    if ContextBuilder is not None:

        try:

            builder = ContextBuilder()

            if hasattr(
                builder,
                "build_context"
            ):
                return builder.build_context(
                    results
                )

            if hasattr(
                builder,
                "create_context"
            ):
                return builder.create_context(
                    results
                )

            if hasattr(
                builder,
                "build"
            ):
                return builder.build(
                    results
                )

        except Exception:
            pass

    # ------------------------------------------------------------
    # Fallback context
    # ------------------------------------------------------------

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        section = result.get(
            "section",
            ""
        )

        title = result.get(
            "title",
            ""
        )

        content = result.get(
            "content",
            ""
        )

        context_parts.append(
            f"""
SOURCE {i}
--------------------------------------------------
Section: {section}
Title: {title}

{content}
"""
        )

    return "\n".join(
        context_parts
    )


# ================================================================
# DISPLAY RETRIEVED SOURCES
# ================================================================

def display_sources(results):
    """
    Display unique legal provisions retrieved from FAISS.
    """

    print()
    print_separator()
    print("RETRIEVED LEGAL SOURCES")
    print_separator()

    seen = set()

    source_number = 1

    for result in results:

        section = result.get(
            "section",
            "Unknown"
        )

        title = result.get(
            "title",
            ""
        )

        score = result.get(
            "score",
            None
        )

        key = (
            str(section),
            str(title)
        )

        if key in seen:
            continue

        seen.add(key)

        if score is not None:

            print(
                f"{source_number}. "
                f"Section {section} — {title} "
                f"(similarity: {score:.4f})"
            )

        else:

            print(
                f"{source_number}. "
                f"Section {section} — {title}"
            )

        source_number += 1

    print()


# ================================================================
# DISPLAY VALIDATION RESULT
# ================================================================

def display_validation(validation):
    """
    Display grounding validation status.

    This does not alter the generated legal answer.
    """

    print()
    print_separator()
    print("GROUNDING VALIDATION")
    print_separator()

    if validation["valid"]:

        print("Status: PASS")
        print()
        print(
            "The generated answer passed the "
            "grounding and format checks."
        )

        return

    print("Status: FAIL")
    print()
    print(
        "WARNING: The generated answer contains "
        "one or more grounding/format issues."
    )

    # ------------------------------------------------------------
    # Section grounding
    # ------------------------------------------------------------

    section_check = validation[
        "section_grounding"
    ]

    if not section_check["valid"]:

        print()
        print("Section grounding: FAIL")

        print(
            "Retrieved sections: "
            f"{section_check['retrieved_sections']}"
        )

        print(
            "Cited sections: "
            f"{section_check['cited_sections']}"
        )

        print(
            "Unsupported sections: "
            f"{section_check['unsupported_sections']}"
        )

    else:

        print(
            "Section grounding: PASS"
        )

    # ------------------------------------------------------------
    # Format
    # ------------------------------------------------------------

    format_check = validation[
        "format"
    ]

    if not format_check["valid"]:

        print()
        print("Output format: FAIL")

        print(
            "Missing headers: "
            f"{format_check['missing_sections']}"
        )

    else:

        print(
            "Output format: PASS"
        )

    # ------------------------------------------------------------
    # Reasoning leakage
    # ------------------------------------------------------------

    reasoning_check = validation[
        "reasoning"
    ]

    if not reasoning_check["valid"]:

        print(
            "Reasoning leakage: FAIL"
        )

    else:

        print(
            "Reasoning leakage: PASS"
        )

    # ------------------------------------------------------------
    # Thinking tags
    # ------------------------------------------------------------

    thinking_check = validation[
        "thinking_tags"
    ]

    if not thinking_check["valid"]:

        print(
            "Thinking tags: FAIL"
        )

    else:

        print(
            "Thinking tags: PASS"
        )

    # ------------------------------------------------------------
    # Placeholders
    # ------------------------------------------------------------

    placeholder_check = validation[
        "placeholders"
    ]

    if not placeholder_check["valid"]:

        print(
            "Template placeholders: FAIL"
        )

    else:

        print(
            "Template placeholders: PASS"
        )

    print()


# ================================================================
# DISPLAY ANSWER
# ================================================================

def display_answer(answer):
    """Display generated legal answer."""

    answer = clean_answer_for_display(
        answer
    )

    print()
    print_separator()
    print("LEGAL RAG ANSWER")
    print_separator()
    print()

    print(answer)

    print()


# ================================================================
# INITIALIZE RETRIEVER
# ================================================================

def initialize_retriever():
    """
    Load FAISS index, metadata and embedding model.
    """

    print(
        "Loading legal retriever..."
    )

    retriever = LegalRetriever(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    print(
        "Retriever ready."
    )

    return retriever


# ================================================================
# INITIALIZE GENERATOR
# ================================================================

def initialize_generator():
    """
    Load local Ollama/Qwen2.5 answer generator.
    """

    print()
    print(
        "Loading local answer generator..."
    )

    generator = LegalAnswerGenerator(
        model_name=OLLAMA_MODEL,
        ollama_url=OLLAMA_URL,
        temperature=0.1
    )

    # ------------------------------------------------------------
    # Check Ollama
    # ------------------------------------------------------------

    if not generator.check_connection():

        raise RuntimeError(
            "\nOllama is not running.\n\n"
            "Please start Ollama and try again.\n"
        )

    # ------------------------------------------------------------
    # Check model
    # ------------------------------------------------------------

    if not generator.check_model():

        raise RuntimeError(
            f"\nOllama model '{OLLAMA_MODEL}' "
            "was not found.\n\n"
            f"Run:\nollama pull {OLLAMA_MODEL}\n"
        )

    print(
        "Answer generator ready."
    )

    return generator


# ================================================================
# INITIALIZE VALIDATOR
# ================================================================

def initialize_validator():
    """
    Initialize the grounding validator.
    """

    print(
        "Loading grounding validator..."
    )

    validator = GroundingValidator()

    print(
        "Grounding validator ready."
    )

    return validator


# ================================================================
# ANSWER QUESTION
# ================================================================

def answer_question(
    question,
    retriever,
    generator,
    validator
):
    """
    Execute the complete RAG pipeline for one question.
    """

    # ------------------------------------------------------------
    # STEP 1 — RETRIEVAL
    # ------------------------------------------------------------

    print()
    print(
        "Searching legal provisions..."
    )

    results = retriever.search(
        query=question,
        top_k=TOP_K
    )

    if not results:

        print(
            "\nNo relevant legal provisions were found."
        )

        return

    print(
        f"Retrieved {len(results)} relevant chunks."
    )

    # ------------------------------------------------------------
    # STEP 2 — CONTEXT BUILDING
    # ------------------------------------------------------------

    print(
        "Building legal context..."
    )

    context = build_context(
        results
    )

    if not context:

        print(
            "\nCould not build legal context."
        )

        return

    print(
        "Context created successfully."
    )

    # ------------------------------------------------------------
    # STEP 3 — LLM GENERATION
    # ------------------------------------------------------------

    print()
    print(
        "Generating legal answer..."
    )

    print(
        "Please wait while Qwen2.5 3B "
        "processes the retrieved legal context..."
    )

    answer = generator.generate_answer(
        question=question,
        context=context
    )

    # ------------------------------------------------------------
    # STEP 4 — CLEAN DISPLAY FORMAT
    # ------------------------------------------------------------

    answer = clean_answer_for_display(
        answer
    )

    # ------------------------------------------------------------
    # STEP 5 — GROUNDING VALIDATION
    # ------------------------------------------------------------

    print(
        "Validating generated answer..."
    )

    validation = validator.validate(
        answer=answer,
        results=results
    )

    # ------------------------------------------------------------
    # STEP 6 — DISPLAY VALIDATION
    # ------------------------------------------------------------

    display_validation(
        validation
    )

    # ------------------------------------------------------------
    # STEP 7 — DISPLAY ANSWER
    # ------------------------------------------------------------

    display_answer(
        answer
    )

    # ------------------------------------------------------------
    # STEP 8 — DISPLAY SOURCES
    # ------------------------------------------------------------

    display_sources(
        results
    )


# ================================================================
# MAIN APPLICATION LOOP
# ================================================================

def main():

    print_header()

    # ------------------------------------------------------------
    # Validate vector store
    # ------------------------------------------------------------

    if not os.path.exists(
        INDEX_PATH
    ):

        print(
            "ERROR: FAISS index was not found."
        )

        print(
            f"\nExpected:\n{INDEX_PATH}"
        )

        return

    if not os.path.exists(
        METADATA_PATH
    ):

        print(
            "ERROR: Metadata file was not found."
        )

        print(
            f"\nExpected:\n{METADATA_PATH}"
        )

        return

    # ------------------------------------------------------------
    # Initialize components
    # ------------------------------------------------------------

    try:

        retriever = initialize_retriever()

        generator = initialize_generator()

        validator = initialize_validator()

    except Exception as e:

        print()
        print("=" * 70)
        print("INITIALIZATION ERROR")
        print("=" * 70)
        print()
        print(str(e))
        print()

        return

    # ------------------------------------------------------------
    # APPLICATION LOOP
    # ------------------------------------------------------------

    while True:

        print()
        print_separator()

        question = input(
            "Enter your legal question:\n> "
        ).strip()

        # --------------------------------------------------------
        # Empty question
        # --------------------------------------------------------

        if not question:

            print(
                "\nPlease enter a legal question."
            )

            continue

        # --------------------------------------------------------
        # Exit commands
        # --------------------------------------------------------

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print()
            print(
                "Thank you for using Legal RAG."
            )
            print()

            break

        # --------------------------------------------------------
        # Process question
        # --------------------------------------------------------

        try:

            answer_question(
                question,
                retriever,
                generator,
                validator
            )

        except KeyboardInterrupt:

            print(
                "\n\nOperation interrupted by user."
            )

            break

        except Exception as e:

            print()
            print_separator()
            print("ERROR")
            print_separator()
            print()
            print(str(e))
            print()

        # --------------------------------------------------------
        # Continue?
        # --------------------------------------------------------

        print_separator()

        choice = input(
            "Ask another question? (y/n): "
        ).strip().lower()

        if choice not in {
            "y",
            "yes"
        }:

            print()
            print(
                "Thank you for using Legal RAG."
            )
            print()

            break


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()