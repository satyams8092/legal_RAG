from typing import Dict, List, Any

from retrieval.retriever import LegalRetriever
from generation.context_builder import build_context, build_sources
from generation.answer_generator import LegalAnswerGenerator


class LegalRAGPipeline:
    """
    Complete Legal RAG pipeline.

    Flow:
        Question
        -> Retriever
        -> Context Builder
        -> LLM
        -> Answer + Sources
    """

    def __init__(
        self,
        top_k: int = 5,
        model: str | None = None
    ):

        self.top_k = top_k

        print("Initializing Legal RAG pipeline...")

        # ----------------------------------------------
        # Retriever
        # ----------------------------------------------

        self.retriever = LegalRetriever()

        # ----------------------------------------------
        # Answer generator
        # ----------------------------------------------

        self.generator = LegalAnswerGenerator(
            model=model
        )

        print("Legal RAG pipeline initialized successfully.")

    # --------------------------------------------------
    # Ask question
    # --------------------------------------------------

    def ask(
        self,
        question: str
    ) -> Dict[str, Any]:

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # ----------------------------------------------
        # Step 1: Retrieval
        # ----------------------------------------------

        results = self.retriever.search(
            query=question,
            top_k=self.top_k
        )

        # ----------------------------------------------
        # Step 2: Build context
        # ----------------------------------------------

        context = build_context(
            results,
            max_results=self.top_k
        )

        # ----------------------------------------------
        # Step 3: Generate answer
        # ----------------------------------------------

        answer = self.generator.generate_answer(
            question=question,
            context=context
        )

        # ----------------------------------------------
        # Step 4: Sources
        # ----------------------------------------------

        sources = build_sources(results)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_results": results
        }