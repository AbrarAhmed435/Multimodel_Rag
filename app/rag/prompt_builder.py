class PromptBuilder:

    def build_prompt(
        self,
        question,
        context
    ):

        prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
say that the information is not available.

Context:

{context}

Question:
{question}

Answer:
"""

        return prompt