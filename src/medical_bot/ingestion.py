from typing import List
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from .config import AppConfig

def load_source_documents(data_dir: str) -> List[Document]:
    """
    Loads all PDF documents from the specified directory.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 20) -> List[Document]:
    """
    Splits documents into smaller chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_embedding_model():
    """
    Initializes and returns the HuggingFace embedding model.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=AppConfig.EMBEDDING_MODEL
    )
    return embeddings
