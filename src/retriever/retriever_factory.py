# src/retrievers/retriever_factory.py

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from typing import List, Optional
from langchain.vectorstores.base import VectorStore

class FaissRetrieverWrapper:
    def __init__(self, vector_store):
        # vector_store: VectorStoreWrapper
        self.vector_store = vector_store
        self.k = 3

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.vector_store.similarity_search(query, k=self.k)


def get_retriever(
    rtype: str,
    doc_texts: List[str],
    vector_store: Optional[VectorStore] = None,
    top_k: int = 3
):
    """Retriever를 생성합니다."""
    
    if rtype == "bm25":
        return BM25Retriever.from_texts(
            doc_texts,
            k=top_k
        )
    elif rtype == "faiss":
        if not vector_store:
            raise ValueError("FAISS retriever requires vector store")
        return vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
    elif rtype == "ensemble":
        if not vector_store:
            raise ValueError("Ensemble retriever requires vector store")
        
        bm25_retriever = BM25Retriever.from_texts(
            doc_texts,
            k=top_k
        )
        faiss_retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        
        return EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
    else:
        raise ValueError(f"Unknown retriever type: {rtype}")
