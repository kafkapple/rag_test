# src/embeddings/embedding_factory.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional
import os


def get_embedding_model(
    etype: str = "openai",
    openai_model: str = "text-embedding-ada-002"
):
    """
    etype: "openai" or "huggingface"
    """
    if etype == "openai":
        # OpenAI API 키 체크
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
        return OpenAIEmbeddings(model=openai_model)
    elif etype == "huggingface":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        raise ValueError(f"Unknown embedding type: {etype}")
