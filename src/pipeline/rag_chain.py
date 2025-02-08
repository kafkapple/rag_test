# src/pipeline/rag_chain.py

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Callable
from langchain.docstore.document import Document
import torch
import warnings

class RAGPipeline:
    """
    Q -> Retriever -> Prompt -> LLM -> Answer
    """

    def __init__(
        self,
        retriever,
        prompt_manager,
        llm_name: str,
        temperature: float = 0.7,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        self.retriever = retriever
        self.prompt_manager = prompt_manager
        self.temperature = temperature  # temperature 저장
        
        # 진행 상황 보고
        if progress_callback:
            progress_callback("토크나이저 초기화 중...", 0.2)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        
        if progress_callback:
            progress_callback("모델 다운로드 중...", 0.4)
            
        # 메모리 최적화 설정
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "load_in_8bit": True,
            "trust_remote_code": True,
            # 메모리 최적화 옵션
            "max_memory": {0: "5GiB"},  # GPU 메모리 제한
            "offload_folder": "offload",
            "offload_state_dict": True,
            "llm_int8_enable_fp32_cpu_offload": True  # CPU 오프로딩 활성화
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            **model_kwargs
        )
        
        if progress_callback:
            progress_callback("파이프라인 설정 중...", 0.8)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            max_new_tokens=1024,     # 256에서 1024로 증가
            min_new_tokens=64,       # 10에서 64로 증가
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1,   # 응답 수
            no_repeat_ngram_size=3    # 반복 방지
        )
        
        if progress_callback:
            progress_callback("초기화 완료!", 1.0)

    def run(self, question: str, style="concise") -> str:
        # 1) retrieve
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            relevant_docs: List[Document] = self.retriever.get_relevant_documents(question)
            
        context = self._format_docs(relevant_docs)
        
        # 2) prompt
        prompt_text = self.prompt_manager.get_prompt(style, context, question)
        
        # 토큰 수 계산
        input_tokens = self.tokenizer(prompt_text, return_length=True)["length"]
        
        # 디버그 출력
        print("\n=== Debug Info ===")
        print(f"Input Token Length: {input_tokens}")
        print(f"Max Context Length: {self.model.config.max_position_embeddings}")
        print(f"Max Output Length: {256}")  # max_new_tokens 값
        print("\nInput Prompt:")
        print("-" * 50)
        print(prompt_text)
        print("-" * 50)

        # 3) generate
        output = self.pipeline(
            prompt_text, 
            max_new_tokens=1024,     # 256에서 1024로 증가
            temperature=self.temperature,
            return_full_text=False
        )
        
        generated_text = output[0]["generated_text"]
        output_tokens = len(self.tokenizer.encode(generated_text))  # 토큰 수 직접 계산
        
        # 디버그 출력
        print("\nModel Response:")
        print(f"Output Token Length: {output_tokens}")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        print("=== End Debug Info ===\n")

        return self._post_process(generated_text, prompt_text)

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
