# src/pipeline/rag_chain.py

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import List
from langchain.docstore.document import Document

class RAGPipeline:
    """
    Q -> Retriever -> Prompt -> LLM -> Answer
    """

    def __init__(self, retriever, prompt_manager, llm_name="gpt2", temperature=0.0):
        self.retriever = retriever
        self.prompt_manager = prompt_manager
        self.llm_name = llm_name
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def run(self, question: str, style="concise") -> str:
        # 1) retrieve
        relevant_docs: List[Document] = self.retriever.get_relevant_documents(question)
        context = self._format_docs(relevant_docs)

        # 2) prompt
        prompt_text = self.prompt_manager.get_prompt(style, context, question)

        # 3) generate
        output = self.pipe(prompt_text, max_new_tokens=128, temperature=self.temperature)[0]["generated_text"]
        return self._post_process(output, prompt_text)

    def _format_docs(self, docs: List[Document]) -> str:
        lines = []
        for doc in docs:
            meta = doc.metadata.get("source") or doc.metadata.get("url") or "unknown"
            lines.append(f"<source: {meta}>\n{doc.page_content}")
        return "\n---\n".join(lines)

    def _post_process(self, full_output: str, prompt_text: str) -> str:
        # remove prompt from the beginning if present
        if full_output.startswith(prompt_text):
            return full_output[len(prompt_text):].strip()
        return full_output.strip()
