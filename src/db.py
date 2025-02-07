"""
db.py

Faiss 기반 Vector DB를 관리하는 단일 모듈.
문서 임베딩 & 인덱스 생성, 로드, 검색.
"""

import os
import pickle
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

class FaissDB:
    """
    단일 책임: Faiss VectorStore 관리.
    1) create_index(docs)
    2) load_index()
    3) similarity_search(query)
    """

    def __init__(self, index_path: str = "./faiss_index/faiss_store.pkl", embeddings=None):
        """
        Args:
            index_path (str): faiss 인덱스를 저장/로드할 경로
            embeddings: LangChain Embedding 객체 (SentenceTransformers, OpenAIEmbeddings 등)
        """
        self.index_path = index_path
        self.embeddings = embeddings
        self.store = None

    def create_index(self, docs: List[Document]):
        """
        docs를 임베딩해 Faiss 인덱스를 생성하고, 로컬 파일로 저장.
        """
        if not docs:
            print("[FaissDB] No documents to index.")
            return
        print("[FaissDB] Creating Faiss index...")
        self.store = FAISS.from_documents(docs, self.embeddings)
        self._save_index()
        print(f"[FaissDB] Created index with {len(docs)} docs.")

    def load_index(self):
        """
        로컬에서 faiss 인덱스 파일을 로드.
        """
        if not os.path.exists(self.index_path):
            print(f"[FaissDB] No index file found at {self.index_path}.")
            return
        with open(self.index_path, "rb") as f:
            self.store = pickle.load(f)
        print(f"[FaissDB] Loaded index from {self.index_path}.")

    def similarity_search(self, query: str, k=3) -> List[Document]:
        """
        Top-k 유사도 검색 결과 반환.
        """
        if not self.store:
            raise ValueError("Faiss store not initialized. Call create_index() or load_index() first.")
        return self.store.similarity_search(query, k=k)

    def get_all_documents(self) -> List[Document]:
        """
        저장된 모든 Document를 반환 (FAISS docstore에서 가져옴).
        """
        if not self.store:
            return []
        docstore = self.store.docstore
        all_docs = []
        for doc_id in docstore._dict.keys():
            doc = docstore.search(doc_id)
            if doc:
                all_docs.append(doc)
        return all_docs

    def _save_index(self):
        """
        self.store(Faiss) 객체를 pickle로 저장.
        """
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.store, f)
        print(f"[FaissDB] Saved index -> {self.index_path}")

class VectorStoreWrapper:
    def __init__(self, vtype: str, persist_dir: str, embeddings):
        self.vtype = vtype
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.vectorstore = None

    def create_from_documents(self, documents: List[Document]):
        if self.vtype == "faiss":
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # 저장
            self.vectorstore.save_local(self.persist_dir)
        else:
            raise ValueError(f"Unsupported vector store type: {self.vtype}")
        
    def as_retriever(self, **kwargs):
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.as_retriever(**kwargs)

    def similarity_search(self, query: str, k: int = 3):
        """유사도 기반 문서 검색"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_all_documents(self):
        """저장된 모든 문서 조회"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.get()
        
    def load_local(self):
        """저장된 벡터 스토어 로드"""
        if os.path.exists(self.persist_dir):
            self.vectorstore = FAISS.load_local(
                self.persist_dir, 
                self.embeddings
            )
            return True
        return False
