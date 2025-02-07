# src/ingestion/splitter.py

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain.docstore.document import Document

def split_documents(
    docs: List[Document],
    splitter_type: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    splitter_type: "character", "recursive", "semantic"
    """
    if splitter_type == "character":
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)

    elif splitter_type == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)

    elif splitter_type == "semantic":
        # SemanticChunker requires an embeddings model
        # ex) OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()  # or config-based
        splitter = SemanticChunker(embeddings, add_start_index=True)
        # semantic splitter returns List[str], so we might need to re-wrap as Document
        splitted_docs = []
        for doc in docs:
            splitted_texts = splitter.split_text(doc.page_content)
            # meta: doc.metadata
            for text_chunk in splitted_texts:
                splitted_docs.append(
                    Document(page_content=text_chunk, metadata=doc.metadata)
                )
        return splitted_docs

    else:
        # fallback
        return docs
