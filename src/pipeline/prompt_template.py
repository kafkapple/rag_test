# src/pipeline/prompt_template.py

class PromptTemplateManager:
    def __init__(self):
        self.templates = {
            "concise": (
                "Answer the question based on the following context.\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            ),
            "detailed": (
                "Answer the question based on the following context.\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            )
        }

    def get_prompt(self, style: str, context: str, question: str) -> str:
        tmpl = self.templates.get(style, self.templates["concise"])
        return tmpl.format(context=context, question=question)
