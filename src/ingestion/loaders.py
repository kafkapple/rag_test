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

def load_web_document(url: str):
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()
    return docs

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
