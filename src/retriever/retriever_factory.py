# src/retrievers/retriever_factory.py

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from typing import List

class FaissRetrieverWrapper:
    def __init__(self, vector_store):
        # vector_store: VectorStoreWrapper
        self.vector_store = vector_store
        self.k = 3

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.vector_store.similarity_search(query, k=self.k)


def get_retriever(rtype: str, doc_texts: List[str] = None, vector_store=None):
    """
    rtype: "bm25", "faiss", "ensemble"
    doc_texts: for BM25
    vector_store: for Faiss
    """
    if rtype == "bm25":
        # BM25Retriever.from_texts(...) 로 생성
        if not doc_texts:
            raise ValueError("doc_texts is required for BM25 retriever.")
        return BM25Retriever.from_texts(doc_texts)

    elif rtype == "faiss":
        if not vector_store:
            raise ValueError("vector_store required for faiss retriever.")
        return vector_store.as_retriever()

    elif rtype == "ensemble":
        # 예시: BM25 + Faiss ensemble
        if not doc_texts or not vector_store:
            raise ValueError("Need doc_texts & vector_store for ensemble.")
        bm25_retriever = BM25Retriever.from_texts(doc_texts)
        faiss_retriever = vector_store.as_retriever()
        return EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

    else:
        raise ValueError(f"Unknown retriever type: {rtype}")
