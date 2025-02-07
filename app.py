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

from src.ingestion.loaders import (
    load_pdf_document, load_txt_document, load_web_document
)
from src.ingestion.splitter import split_documents
from src.embeddings.embedding_factory import get_embedding_model
from src.db import VectorStoreWrapper
from src.retriever.retriever_factory import get_retriever

from src.pipeline.prompt_template import PromptTemplateManager
from src.pipeline.rag_chain import RAGPipeline

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
    pdf_docs = []
    web_docs = []
    # 예시: PDF
    pdf_path = st.text_input("PDF path:", "./data/example.pdf")
    if st.button("Load PDF"):
        pdf_docs = load_pdf_document(pdf_path)
        st.write(f"Loaded {len(pdf_docs)} pages from PDF.")
    # 예시: 웹
    web_url = st.text_input("Web URL:", "https://huggingface.co/docs")
    if st.button("Load Web"):
        web_docs = load_web_document(web_url)
        st.write(f"Loaded {len(web_docs)} docs from Web page.")

    docs = pdf_docs + web_docs
    st.write(f"Total raw docs: {len(docs)}")

    # ----------------------------
    # 2) Split Documents
    # ----------------------------
    st.header("2) Split Documents")
    if st.button("Split Docs"):
        splitted_docs = split_documents(
            docs,
            splitter_type=cfg.splitter.type,
            chunk_size=cfg.splitter.chunk_size,
            chunk_overlap=cfg.splitter.chunk_overlap
        )
        st.write(f"Splitted docs: {len(splitted_docs)}")
        # store in session_state
        st.session_state["splitted_docs"] = splitted_docs

    # ----------------------------
    # 3) Embeddings & VectorStore
    # ----------------------------
    st.header("3) Create Vector Store")
    if "splitted_docs" not in st.session_state:
        st.warning("Please split docs first.")
    else:
        splitted_docs = st.session_state["splitted_docs"]
        if st.button("Create VectorStore"):
            # init embeddings
            emb_model = get_embedding_model(
                etype=cfg.embeddings.type,
                openai_model=cfg.embeddings.openai_model
            )
            vstore = VectorStoreWrapper(
                vtype=cfg.vectorstore.type,
                persist_dir=cfg.vectorstore.persist_dir,
                embeddings=emb_model
            )
            vstore.create_from_documents(splitted_docs)
            st.session_state["vector_store"] = vstore
            st.success("VectorStore created & saved.")

    # ----------------------------
    # 4) Retriever
    # ----------------------------
    st.header("4) Create Retriever (BM25 / FAISS / Ensemble)")
    if st.button("Setup Retriever"):
        if "vector_store" not in st.session_state:
            st.warning("Create vector store first.")
        else:
            vstore = st.session_state["vector_store"]
            # for BM25 we need raw texts
            raw_texts = [doc.page_content for doc in splitted_docs]
            retriever = get_retriever(
                rtype=cfg.retriever.type,
                doc_texts=raw_texts,
                vector_store=vstore
            )
            # set top_k
            if hasattr(retriever, "k"):
                retriever.k = cfg.retriever.top_k
            st.session_state["retriever"] = retriever
            st.success(f"Retriever {cfg.retriever.type} is ready.")

    # ----------------------------
    # 5) RAG Pipeline
    # ----------------------------
    st.header("5) RAG QA")
    query = st.text_input("Your question:")
    if st.button("Get Answer"):
        if "retriever" not in st.session_state:
            st.warning("Please setup retriever first.")
        else:
            ret = st.session_state["retriever"]
            prompt_mgr = PromptTemplateManager()
            rag = RAGPipeline(
                retriever=ret,
                prompt_manager=prompt_mgr,
                llm_name=cfg.llm.model,
                temperature=cfg.llm.temperature
            )
            answer = rag.run(query, style=cfg.prompt.style)
            st.write("### Answer")
            st.write(answer)

    # ----------------------------
    # 6) Document Search
    # ----------------------------
    st.header("6) Search Documents")
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
