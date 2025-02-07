# My RAG Project (PDF/Web + Korean/English)

본 프로젝트는 LangChain을 활용해 **PDF / 웹 문서**를 인덱싱하여, 로컬 LLM과 연동한 RAG QA 시스템을 예시로 제공합니다.

## 주요 기능
- **PDF/웹 문서 로더**: `ingestion.py`
  - PDFIngestion: PyPDFLoader로 한글/영문 PDF 로딩
  - Advanced Web Crawler: 링크 재귀 탐색
  - 한글 문단 기준 `RecursiveCharacterTextSplitter` 예시
- **벡터 DB**: `vectordb.py`
  - Faiss 기반, 로컬 인덱싱/검색
- **Prompt Template**: `prompt_template.py`
  - 간결/상세 등 원하는 QA 스타일 조절
- **RAG Pipeline**: `rag_pipeline.py`
  - 검색→프롬프트→LLM생성→응답
- **Streamlit UI**: `app.py`
  - 사용자가 쿼리 입력 & 응답 확인

## 설치 & 실행
1. Conda 환경 설정
   ```bash
   conda env create -f environment.yml
   conda activate my-rag-project
