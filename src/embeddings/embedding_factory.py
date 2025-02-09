# src/embeddings/embedding_factory.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_embedding_model(provider: str, model_id: str):
    """임베딩 모델을 생성합니다."""
    
    if provider == "huggingface":
        # HuggingFace 임베딩 설정
        model_kwargs = {
            'device': 'cpu',  # CUDA 오류 방지를 위해 CPU 사용
            'token': os.getenv("HUGGINGFACE_TOKEN")
        }
        encode_kwargs = {
            'normalize_embeddings': True
        }
        return HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    elif provider == "openai":
        return OpenAIEmbeddings(
            model=model_id,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"지원하지 않는 임베딩 제공자입니다: {provider}")
