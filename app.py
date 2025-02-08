"""
app.py

Streamlit + Hydra + dotenv
모든 모듈 조립하여 최종 RAG QA.
Ensemble Retriever 시연.
Semantic Splitter 시연.
"""
import os
import streamlit as st
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
import time
import traceback

from src.ingestion.loaders import (
    load_pdf_document, load_txt_document, load_web_document
)
from src.ingestion.splitter import split_documents
from src.embeddings.embedding_factory import get_embedding_model
from src.db import VectorStoreWrapper
from src.retriever.retriever_factory import get_retriever

from src.pipeline.prompt_template import PromptTemplateManager
from src.pipeline.rag_chain import RAGPipeline

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다. "
        ".env 파일을 확인하거나 환경 변수를 설정해주세요."
    )
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "./models/"
# Streamlit 설정
st.set_page_config(
    page_title="RAG Pipeline Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 파일 감시 비활성화
if not st.session_state.get("watched_files"):
    st.session_state["watched_files"] = True
    import streamlit.watcher.path_watcher
    streamlit.watcher.path_watcher.watch_file = lambda x: None

def display_split_documents():
    """분할된 문서를 표시하는 함수"""
    if "docs_by_source" not in st.session_state:
        return
        
    st.write("### 분할된 문서 조회")
    docs_by_source = st.session_state["docs_by_source"]
    
    for source, source_docs in docs_by_source.items():
        st.subheader(f"📄 {source} ({len(source_docs)} chunks)")
        
        # 페이지네이션
        chunks_per_page = 5
        total_pages = len(source_docs) // chunks_per_page + (1 if len(source_docs) % chunks_per_page else 0)
        
        # source에서 유효한 key 문자열 생성
        safe_key = "".join(c if c.isalnum() else "_" for c in source)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                key=f"page_{safe_key}"
            )
        with col2:
            st.write(f"Total: {total_pages} pages")
        
        start_idx = (page - 1) * chunks_per_page
        end_idx = min(start_idx + chunks_per_page, len(source_docs))
        
        # 탭으로 청크 표시
        chunk_tabs = st.tabs([f"Chunk {i+1}" for i in range(start_idx, end_idx)])
        for tab, i in zip(chunk_tabs, range(start_idx, end_idx)):
            with tab:
                doc = source_docs[i]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**내용:**")
                    st.markdown(doc.page_content)
                with col2:
                    st.markdown("**메타데이터:**")
                    st.json(doc.metadata)
        
        st.markdown("---")  # 소스 구분선

def main():
    # Hydra 재초기화
    GlobalHydra.instance().clear()
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name="config")
    
    load_dotenv()
    st.title(cfg.app.project_name)

    # ----------------------------
    # 1) Load Documents
    # ----------------------------
    st.header("1) Load Documents")
    
    # 초기화
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []
    if "web_docs" not in st.session_state:
        st.session_state.web_docs = []
    
    # 예시: PDF
    pdf_path = st.text_input("PDF path:", cfg.paths.pdf_path)
    if st.button("Load PDF"):
        st.session_state.pdf_docs = load_pdf_document(pdf_path)
        st.write(f"Loaded {len(st.session_state.pdf_docs)} pages from PDF.")
    
    # 예시: 웹
    web_url = st.text_input("Web URL:", cfg.crawler.start_url)
    if st.button("Load Web"):
        st.session_state.web_docs = load_web_document(web_url)
        st.write(f"Loaded {len(st.session_state.web_docs)} docs from Web page.")

    # 전체 문서 합치기
    docs = st.session_state.pdf_docs + st.session_state.web_docs
    st.write(f"Total raw docs: {len(docs)}")

    # ----------------------------
    # 2) Split Documents
    # ----------------------------
    st.header("2) Split Documents")
    
    # Splitter 옵션 설정
    st.subheader("Splitter 설정")
    col1, col2, col3 = st.columns(3)
    with col1:
        splitter_type = st.selectbox(
            "Splitter 타입",
            options=["character", "token", "markdown", "recursive"],
            help="문서 분할 방식을 선택하세요"
        )
    with col2:
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=cfg.splitter.chunk_size,
            help="각 청크의 최대 길이"
        )
    with col3:
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=chunk_size-1,
            value=cfg.splitter.chunk_overlap,
            help="청크 간 중복되는 길이"
        )
    
    if st.button("Split Docs"):
        if not docs:
            st.warning("문서를 먼저 로드해주세요.")
            st.write("### 디버그 정보:")
            st.write(f"PDF 문서 수: {len(st.session_state.pdf_docs)}")
            st.write(f"웹 문서 수: {len(st.session_state.web_docs)}")
            return
            
        try:
            st.write("### 문서 분할 시작")
            st.write(f"입력 문서 수: {len(docs)}")
            
            splitted_docs = split_documents(
                docs,
                splitter_type=splitter_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            if not splitted_docs:
                st.error("문서 분할 결과가 비어있습니다.")
                return
                
            st.write(f"분할된 문서 수: {len(splitted_docs)}")
            
            # 문서를 소스별로 그룹화
            docs_by_source = {}
            for doc in splitted_docs:
                source = doc.metadata.get('source', 'unknown')
                if source not in docs_by_source:
                    docs_by_source[source] = []
                docs_by_source[source].append(doc)
            
            # session_state에 저장
            st.session_state["splitted_docs"] = splitted_docs
            st.session_state["docs_by_source"] = docs_by_source
            st.success("문서 분할이 완료되었습니다.")
            
        except Exception as e:
            st.error(f"문서 분할 중 오류가 발생했습니다: {str(e)}")
            st.write("### 디버그 정보:")
            st.write(f"문서 타입: {type(docs)}")
            st.write(f"설정 정보:")
            st.write(f"- splitter_type: {splitter_type}")
            st.write(f"- chunk_size: {chunk_size}")
            st.write(f"- chunk_overlap: {chunk_overlap}")
            st.code(traceback.format_exc())

    # 문서가 분할되어 있을 때만 한 번 표시
    if "docs_by_source" in st.session_state and not st.session_state.get("display_triggered"):
        display_split_documents()
        st.session_state["display_triggered"] = True

    # ----------------------------
    # 3) Embeddings & VectorStore
    # ----------------------------
    st.header("3) Create Vector Store")
    
    # 임베딩 설정
    st.subheader("임베딩 설정")
    emb_providers = ["huggingface", "openai"]
    emb_provider = st.selectbox(
        "임베딩 제공자",
        options=emb_providers,
        index=emb_providers.index(cfg.embeddings.provider)
    )
    
    if emb_provider == "huggingface":
        hf_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "jhgan/ko-sroberta-multitask",
        ]
        emb_model_id = st.selectbox(
            "Hugging Face 임베딩 모델",
            options=hf_models,
            index=hf_models.index(cfg.embeddings.model_configs.huggingface.model_id)
        )
    else:
        openai_models = ["text-embedding-ada-002"]
        emb_model_id = st.selectbox(
            "OpenAI 임베딩 모델",
            options=openai_models,
            index=openai_models.index(cfg.embeddings.model_configs.openai.model_id)
        )

    if "splitted_docs" not in st.session_state:
        st.warning("문서를 먼저 분할해주세요.")
    else:
        splitted_docs = st.session_state["splitted_docs"]
        if st.button("Create VectorStore"):
            try:
                st.write("### 벡터 저장소 생성 시작")
                # init embeddings
                st.write(f"임베딩 모델 초기화 (provider: {emb_provider}, model: {emb_model_id})")
                
                # 환경 변수 확인
                if emb_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
                    st.error("OpenAI API 키가 설정되지 않았습니다.")
                    return
                if emb_provider == "huggingface" and not os.getenv("HUGGINGFACE_TOKEN"):
                    st.error("Hugging Face 토큰이 설정되지 않았습니다.")
                    return
                
                emb_model = get_embedding_model(
                    provider=emb_provider,
                    model_id=emb_model_id
                )
                
                # 임베딩 테스트
                try:
                    st.write("임베딩 테스트 수행 중...")
                    test_text = splitted_docs[0].page_content
                    st.write(f"테스트 텍스트 미리보기: {test_text[:100]}...")
                    test_embedding = emb_model.embed_query(test_text)
                    st.write(f"임베딩 차원: {len(test_embedding)}")
                except Exception as e:
                    st.error(f"임베딩 테스트 중 오류 발생: {str(e)}")
                    st.write("### 디버그 정보:")
                    st.write(f"- HuggingFace 토큰: {'설정됨' if os.getenv('HUGGINGFACE_TOKEN') else '설정되지 않음'}")
                    st.write(f"- 모델: {emb_model_id}")
                    st.code(traceback.format_exc())
                    return
                
                if not test_embedding:
                    st.error("임베딩 생성에 실패했습니다.")
                    return
                
                st.write("벡터 저장소 초기화 중...")
                vstore = VectorStoreWrapper(
                    vtype=cfg.vectorstore.type,
                    persist_dir=cfg.vectorstore.persist_dir,
                    embeddings=emb_model
                )
                
                st.write(f"문서 {len(splitted_docs)}개를 벡터 저장소에 추가하는 중...")
                vstore.create_from_documents(splitted_docs)
                st.session_state["vector_store"] = vstore
                st.success("벡터 저장소가 생성되었습니다.")
                
            except Exception as e:
                st.error(f"벡터 저장소 생성 중 오류가 발생했습니다: {str(e)}")
                st.write("### 디버그 정보:")
                st.write(f"분할된 문서 수: {len(splitted_docs)}")
                st.write(f"설정 정보:")
                st.write(f"- vectorstore type: {cfg.vectorstore.type}")
                st.write(f"- persist_dir: {cfg.vectorstore.persist_dir}")
                st.write(f"- embedding provider: {emb_provider}")
                st.write(f"- embedding model: {emb_model_id}")
                st.code(traceback.format_exc())

    # ----------------------------
    # 4) LLM 설정
    # ----------------------------
    st.header("4) LLM 설정")
    llm_providers = ["huggingface", "openai"]
    llm_provider = st.selectbox(
        "LLM 제공자",
        options=llm_providers,
        index=llm_providers.index(cfg.llm.provider)
    )
    
    if llm_provider == "huggingface":
        hf_llm_models = [
            "Qwen/Qwen1.5-0.5B-Chat",  # 0.5B 모델
            "Qwen/Qwen2.5-7B-Instruct",  # 1.8B 모델
        ]
        llm_model_id = st.selectbox(
            "Hugging Face 모델",
            options=hf_llm_models,
            index=hf_llm_models.index(cfg.llm.model_configs.huggingface.model_id)
        )
    else:
        openai_llm_models = ["gpt-3.5-turbo", "gpt-4"]
        llm_model_id = st.selectbox(
            "OpenAI 모델",
            options=openai_llm_models,
            index=openai_llm_models.index(cfg.llm.model_configs.openai.model_id)
        )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=cfg.llm.model_configs[llm_provider].temperature,
        step=0.1
    )

    # ----------------------------
    # 5) Retriever
    # ----------------------------
    st.header("5) Create Retriever")
    
    # Retriever 설정
    st.subheader("Retriever 설정")
    retriever_types = ["bm25", "faiss", "ensemble"]
    retriever_type = st.selectbox(
        "Retriever 타입",
        options=retriever_types,
        index=retriever_types.index(cfg.retriever.type),
        help="문서 검색 방식을 선택하세요"
    )
    
    top_k = st.number_input(
        "검색 문서 수 (Top K)",
        min_value=1,
        max_value=20,
        value=cfg.retriever.top_k,
        help="검색할 관련 문서의 수"
    )

    if st.button("Setup Retriever"):
        if "vector_store" not in st.session_state:
            st.warning("Create vector store first.")
        else:
            try:
                vstore = st.session_state["vector_store"]
                # for BM25 we need raw texts
                raw_texts = [doc.page_content for doc in st.session_state["splitted_docs"]]
                
                retriever = get_retriever(
                    rtype=retriever_type,
                    doc_texts=raw_texts,
                    vector_store=vstore,
                    top_k=top_k
                )
                
                st.session_state["retriever"] = retriever
                st.success(f"Retriever ({retriever_type}) 설정이 완료되었습니다.")
                
            except Exception as e:
                st.error(f"Retriever 설정 중 오류 발생: {str(e)}")
                st.code(traceback.format_exc())

    # ----------------------------
    # 6) RAG Pipeline
    # ----------------------------
    st.header("6) RAG QA")
    
    # 프롬프트 스타일 선택
    st.subheader("프롬프트 설정")
    prompt_mgr = PromptTemplateManager()
    prompt_styles = list(prompt_mgr.templates.keys())
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_style = st.selectbox(
            "프롬프트 스타일",
            options=prompt_styles,
            index=prompt_styles.index("concise") if "concise" in prompt_styles else 0,
            help="답변 생성 스타일을 선택하세요"
        )
    
    # 선택된 템플릿 미리보기
    with col2:
        st.text_area(
            "템플릿 미리보기",
            value=prompt_mgr.templates[selected_style],
            height=150,
            disabled=True
        )
    
    # 질문 입력
    query = st.text_input("Your question:")
    if st.button("Get Answer"):
        if "retriever" not in st.session_state:
            st.warning("Please setup retriever first.")
            return
            
        ret = st.session_state["retriever"]
        
        # 검색된 문서 먼저 표시
        st.write("### Retrieved Documents")
        relevant_docs = ret.get_relevant_documents(query)
        for i, doc in enumerate(relevant_docs, 1):
            with st.expander(f"Document {i}"):
                st.write(doc.page_content)
                st.write(f"Source: {doc.metadata.get('source', 'unknown')}")
        
        # 모델 초기화 및 답변 생성
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        def progress_callback(step: str, progress: float):
            progress_bar.progress(progress)
            progress_text.write(f"🤖 {step}")
        
        with st.spinner("Generating answer..."):
            try:
                rag = RAGPipeline(
                    retriever=ret,
                    prompt_manager=prompt_mgr,
                    llm_name=cfg.llm.model_configs[llm_provider].model_id,
                    temperature=temperature,
                    progress_callback=progress_callback
                )
                
                answer = rag.run(query, style=selected_style)
                
                # 진행 상태 표시 제거
                progress_text.empty()
                progress_bar.empty()
                
                st.write("### Answer")
                st.markdown(answer)
                
                # 디버그 정보 표시
                with st.expander("Debug Info"):
                    st.text("Check the terminal/console for detailed debug output")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.code(traceback.format_exc())

    # ----------------------------
    # 7) Document Search
    # ----------------------------
    st.header("7) Search Documents")
    search_query = st.text_input("Search query:")
    k = st.slider("Number of documents", 1, 10, 3)
    
    if st.button("Search"):
        if "vector_store" not in st.session_state:
            st.warning("Please create vector store first.")
        else:
            vstore = st.session_state["vector_store"]
            docs = vstore.similarity_search(search_query, k=k)
            
            st.write("### Search Results")
            for i, doc in enumerate(docs, 1):
                st.write(f"**Document {i}**")
                st.write(doc.page_content)
                st.write("---")

if __name__ == "__main__":
    main()
