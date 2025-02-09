# RAG Pipeline Demo


## 주요 기능
- **PDF/웹 문서 로더, 문서 분할기**: `ingestion/`
  - `loaders.py`: PDF, 웹 문서 로딩
  - `splitter.py`: 문서 분할
- **Embeddings**: `embeddings/`
  - `embeddings_factory.py`: Embeddings 팩토리
- **벡터 DB**: 
  - `db.py`: Faiss 기반, 로컬 인덱싱/검색
- **Retriever**: `retriever/`
  - `retriever_factory.py`: 검색 쿼리 처리
- **Pipeline**: `pipeline/`
  - 검색→프롬프트→LLM생성→응답
  - `prompt_template.py`: 프롬프트 템플릿
  - `rag_chain.py`: 검색→프롬프트→LLM생성→응답
- **Streamlit UI**: `app.py`
  - 사용자가 쿼리 입력 & 응답 확인

## 설치 & 실행
1. Conda 환경 설정
   ```bash
   conda env create -f environment.yml
   conda activate rag
   ```

2. 환경 변수 설정
   - 프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
   - 또는 터미널에서 직접 설정:
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```
