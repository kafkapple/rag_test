# src/ingestion/loaders.py

import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader
)
from typing import List
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
import streamlit as st

def load_web_document(url: str) -> List[Document]:
    """웹 페이지를 로드합니다."""
    try:
        response = requests.get(url)
        # 응답 헤더에서 인코딩 확인
        if 'charset' in response.headers.get('content-type', '').lower():
            response.encoding = response.headers['content-type'].split('charset=')[-1]
        else:
            # 기본값으로 UTF-8 설정
            response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.content.decode(response.encoding, 'replace'), 
                           'html.parser')
        
        # 불필요한 태그 제거
        for tag in soup(['script', 'style', 'head']):
            tag.decompose()
        
        # 텍스트 추출 및 정리
        text = ' '.join(soup.stripped_strings)
        text = ' '.join(text.split())  # 중복 공백 제거
        
        return [
            Document(
                page_content=text,
                metadata={"source": url}
            )
        ]
    except Exception as e:
        st.error(f"웹 페이지 로드 중 오류 발생: {str(e)}")
        return []

def load_pdf_document(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs

def load_csv_document(csv_path: str):
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    return docs

def load_txt_document(txt_path: str):
    loader = TextLoader(txt_path)
    docs = loader.load()
    return docs

def load_directory(dir_path: str, glob_pattern="*.txt"):
    loader = DirectoryLoader(dir_path, glob=glob_pattern, show_progress=True)
    docs = loader.load()
    return docs
