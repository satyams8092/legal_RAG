"""
Local LLM Answer Generator
Legal RAG - Step 9

Uses Ollama + Qwen2.5 3B for local legal answer generation.

Requirements:
    - Ollama must be running
    - qwen2.5:3b must be installed

Ollama endpoint:
    http://localhost:11434
"""

import json
import re
import urllib.request
import urllib.error


class LegalAnswerGenerator:
    """
    Generates grounded legal answers using a local Ollama model.
    """

    def __init__(
        self,
        model_name="qwen2.5:3b",
        ollama_url="http://localhost:11434",
        temperature=0.1
    ):
        self.model_name = model_name

        # Backward compatibility
        self.model = model_name

        self.ollama_url = ollama_url.rstrip("/")
        self.temperature = temperature

        print("Answer generator loaded successfully.")
        print(f"Model: {self.model_name}")
        print(f"Ollama: {self.ollama_url}")

    # ============================================================
    # OLLAMA CONNECTION
    # ============================================================

    def check_connection(self):
        """Check whether Ollama is running."""

        url = f"{self.ollama_url}/api/tags"

        try:
            request = urllib.request.Request(
                url,
                method="GET"
            )

            with urllib.request.urlopen(
                request,
                timeout=10
            ) as response:

                return response.status == 200

        except Exception:
            return False

    # ============================================================
    # CHECK MODEL
    # ============================================================

    def check_model(self):
        """Check whether the required Ollama model is installed."""

        url = f"{self.ollama_url}/api/tags"

        try:
            request = urllib.request.Request(
                url,
                method="GET"
            )

            with urllib.request.urlopen(
                request,
                timeout=10
            ) as response:

                data = json.loads(
                    response.read().decode("utf-8")
                )

            models = data.get("models", [])

            for model in models:

                if model.get("name") == self.model_name:
                    return True

            return False

        except Exception:
            return False

    # ============================================================
    # SYSTEM PROMPT
    # ============================================================

    def build_system_prompt(self):
        """
        System prompt for grounded legal answering.
        """

        return """
You are a legal question-answering assistant.

You are part of a Retrieval-Augmented Generation system.

Your job is ONLY to answer the user's question using the
LEGAL CONTEXT provided in the user message.

STRICT RULES:

1. Use ONLY the supplied legal context.
2. Do NOT use outside knowledge.
3. Do NOT invent legal provisions.
4. Do NOT invent section numbers.
5. Do NOT invent definitions.
6. Do NOT invent penalties.
7. Do NOT invent rights, duties, procedures, or powers.
8. Do NOT guess.
9. If the context does not contain enough information, explicitly say so.
10. Mention the relevant section number and title when available.
11. Answer the question directly.
12. Keep the answer concise and legally precise.
13. Do NOT explain your reasoning.
14. Do NOT describe how you analyzed the context.
15. Do NOT describe your search process.
16. Do NOT mention SOURCE 1, SOURCE 2, etc.
17. Do NOT repeat the entire legal context.
18. Do NOT create a draft before the final answer.
19. Do NOT use phrases such as:
    "Let's analyze"
    "Let's check"
    "Let's extract"
    "I need to answer"
    "The user asks"
    "We need to"
    "First, I will"
    "Step 1"
    "Step 2"
    "Looking at the context"
20. Do NOT output internal reasoning.

If the supplied context does not contain sufficient information
to answer the question, use this exact sentence:

The retrieved legal context does not contain sufficient information to answer this question.

OUTPUT FORMAT:

Answer:
[direct answer]

Relevant Sections:
[section number and title]

Explanation:
[brief explanation based ONLY on the supplied context]

Sources:
[section number(s)]

IMPORTANT:

Return ONLY the final answer.

Do not return reasoning.
Do not return analysis.
Do not return a draft.
Do not return instructions.
Do not repeat the prompt.
"""

    # ============================================================
    # CLEAN RESPONSE
    # ============================================================

    def clean_response(self, text):
        """
        Clean the final Ollama response.

        Removes:
        - Qwen thinking blocks
        - markdown fences
        - reasoning prefixes
        - duplicate output
        - prompt/template leakage
        """

        if not text:
            return ""

        # --------------------------------------------------------
        # Remove <think>...</think>
        # --------------------------------------------------------

        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Remove unclosed <think> blocks if present
        text = re.sub(
            r"<think>.*$",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

        text = text.strip()

        if not text:
            return ""

        # --------------------------------------------------------
        # Remove markdown code fences
        # --------------------------------------------------------

        text = re.sub(
            r"^```(?:text|markdown)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        text = text.strip()

        # --------------------------------------------------------
        # Find the final Answer section.
        #
        # If the model first repeats the template and then gives
        # the actual answer, use the final Answer section.
        # --------------------------------------------------------

        answer_matches = list(
            re.finditer(
                r"(?im)^\s*Answer\s*:\s*$",
                text
            )
        )

        if answer_matches:

            # Prefer the final Answer section because smaller
            # local models sometimes repeat the template first.
            text = text[
                answer_matches[-1].start():
            ].strip()

        # --------------------------------------------------------
        # Remove obvious reasoning lines appearing before the
        # actual structured output.
        # --------------------------------------------------------

        reasoning_patterns = [
            r"^\s*First,\s+I need to",
            r"^\s*First,\s+we need to",
            r"^\s*We need to",
            r"^\s*I need to answer",
            r"^\s*The user asks",
            r"^\s*Let's analyze",
            r"^\s*Let's check",
            r"^\s*Let's look",
            r"^\s*Let's extract",
            r"^\s*Looking at the context",
            r"^\s*Based on the context, I",
            r"^\s*I should",
            r"^\s*We must",
            r"^\s*Step\s+\d+\s*:",
        ]

        lines = text.splitlines()

        cleaned_lines = []

        for line in lines:

            stripped = line.strip()

            if not stripped:
                cleaned_lines.append(line)
                continue

            is_reasoning = any(
                re.match(
                    pattern,
                    stripped,
                    flags=re.IGNORECASE
                )
                for pattern in reasoning_patterns
            )

            if is_reasoning:
                continue

            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines).strip()

        # --------------------------------------------------------
        # Remove template placeholders.
        #
        # Smaller models sometimes output:
        #
        # [direct answer]
        # [section number and title]
        #
        # before producing the actual response.
        # --------------------------------------------------------

        placeholder_patterns = [
            r"^\[direct answer\]\s*$",
            r"^\[section number and title\]\s*$",
            r"^\[brief explanation based only on the supplied context\]\s*$",
            r"^\[section number\(s\)\]\s*$",
            r"^\[section number\(s\)\]\s*$",
        ]

        filtered_lines = []

        for line in text.splitlines():

            stripped = line.strip()

            if any(
                re.fullmatch(
                    pattern,
                    stripped,
                    flags=re.IGNORECASE
                )
                for pattern in placeholder_patterns
            ):
                continue

            filtered_lines.append(line)

        text = "\n".join(filtered_lines).strip()

        # --------------------------------------------------------
        # Remove duplicate Answer headers
        # --------------------------------------------------------

        text = re.sub(
            r"(?im)^\s*Answer\s*:\s*$",
            "Answer:",
            text
        )

        # --------------------------------------------------------
        # Normalize section headers
        # --------------------------------------------------------

        text = re.sub(
            r"(?im)^\s*Relevant Sections\s*:\s*$",
            "Relevant Sections:",
            text
        )

        text = re.sub(
            r"(?im)^\s*Explanation\s*:\s*$",
            "Explanation:",
            text
        )

        text = re.sub(
            r"(?im)^\s*Sources\s*:\s*$",
            "Sources:",
            text
        )

        # --------------------------------------------------------
        # Remove excessive blank lines
        # --------------------------------------------------------

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # ============================================================
    # GENERATE ANSWER
    # ============================================================

    def generate_answer(
        self,
        question,
        context
    ):
        """
        Generate a grounded legal answer using Ollama.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not context or not context.strip():
            raise ValueError(
                "Legal context cannot be empty."
            )

        system_prompt = self.build_system_prompt()

        user_prompt = f"""
LEGAL CONTEXT
============

{context}

END LEGAL CONTEXT
=================


USER QUESTION
=============

{question}


Now answer the question.

Use ONLY the legal context.

If the answer is not contained in the context, say:

The retrieved legal context does not contain sufficient information to answer this question.

Return ONLY the following format:

Answer:
...

Relevant Sections:
...

Explanation:
...

Sources:
...
"""

        payload = {
            "model": self.model_name,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            "stream": False,

            # Disable reasoning where supported.
            "think": False,

            "options": {
                # Very low temperature improves consistency
                # for legal extraction.
                "temperature": self.temperature,

                # Context suitable for 8 GB RAM.
                "num_ctx": 4096,

                # Enough room for a concise legal answer.
                "num_predict": 400,

                # Suitable for the available CPU/RAM.
                "num_thread": 4
            }
        }

        url = f"{self.ollama_url}/api/chat"

        data = json.dumps(
            payload
        ).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=600
            ) as response:

                raw_response = response.read().decode(
                    "utf-8"
                )

        except urllib.error.HTTPError as e:

            error_body = e.read().decode(
                "utf-8",
                errors="ignore"
            )

            raise RuntimeError(
                f"Ollama HTTP error {e.code}: "
                f"{error_body}"
            )

        except urllib.error.URLError as e:

            raise RuntimeError(
                f"Could not connect to Ollama: {e}"
            )

        except TimeoutError:

            raise RuntimeError(
                "Ollama generation timed out."
            )

        # --------------------------------------------------------
        # Parse JSON
        # --------------------------------------------------------

        try:

            response_data = json.loads(
                raw_response
            )

        except json.JSONDecodeError:

            raise RuntimeError(
                "Ollama returned invalid JSON."
            )

        # --------------------------------------------------------
        # Extract assistant message
        # --------------------------------------------------------

        message = response_data.get(
            "message",
            {}
        )

        answer = message.get(
            "content",
            ""
        )

        if not answer:

            raise RuntimeError(
                "Ollama returned an empty answer."
            )

        # --------------------------------------------------------
        # Clean answer
        # --------------------------------------------------------

        answer = self.clean_response(
            answer
        )

        if not answer:

            raise RuntimeError(
                "The generated answer was empty "
                "after cleaning."
            )

        return answer


# ================================================================
# FACTORY FUNCTION
# ================================================================

def create_answer_generator(
    model_name="qwen2.5:3b",
    ollama_url="http://localhost:11434",
    temperature=0.1
):
    """
    Factory function for creating the answer generator.
    """

    return LegalAnswerGenerator(
        model_name=model_name,
        ollama_url=ollama_url,
        temperature=temperature
    )