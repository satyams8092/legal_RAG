from typing import List, Dict


def build_context(
    results: List[Dict],
    max_results: int = 5
) -> str:
    """
    Convert retriever results into a clean context
    that can later be passed to an LLM.
    """

    if not results:
        return "No relevant legal provisions were found."

    context_parts = []

    for i, result in enumerate(results[:max_results], start=1):

        section = result.get("section", "")
        title = result.get("title", "")
        chapter = result.get("chapter", "")
        content = result.get("content", "")

        part = f"""
SOURCE {i}
Section: {section}
Title: {title}
Chapter: {chapter}

Provision:
{content}
""".strip()

        context_parts.append(part)

    return "\n\n" + "\n\n".join(
        ["=" * 70, *context_parts, "=" * 70]
    )


def build_sources(results: List[Dict]) -> List[Dict]:
    """
    Extract source information separately so that
    the final answer can display legal citations.
    """

    sources = []

    for result in results:

        sources.append({
            "section": result.get("section", ""),
            "title": result.get("title", ""),
            "chapter": result.get("chapter", ""),
            "score": result.get("score", 0.0),
            "chunk_id": result.get("chunk_id", "")
        })

    return sources


if __name__ == "__main__":
    print("Context builder module loaded successfully.")