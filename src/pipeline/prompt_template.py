# src/pipeline/prompt_template.py

class PromptTemplateManager:
    def __init__(self):
        self.templates = {
            "concise": (
                "다음 컨텍스트를 참고하여 질문에 간결하게 답해주세요.\n"
                "컨텍스트:\n{context}\n\n"
                "질문: {question}\n\n"
                "답변:"
            ),
            "detailed": (
                "아래 컨텍스트를 기반으로 질문에 대해 상세히 답해주세요.\n"
                "컨텍스트:\n{context}\n\n"
                "질문: {question}\n\n"
                "답변:"
            )
        }

    def get_prompt(self, style: str, context: str, question: str) -> str:
        tmpl = self.templates.get(style, self.templates["concise"])
        return tmpl.format(context=context, question=question)
