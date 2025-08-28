from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


def load_document(file_path: str) -> List[Document]:
    """
    Load a document into memory depending on file type.
    Supports PDF, TXT, DOCX.
    Returns a list of LangChain Document objects.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    return loader.load()

def chunk_documents(docs: str, chunk_size: int = 500, overlap: int = 50) -> List[Document]:
    """
    Split large documents into smaller chunks so the LLM can handle them.
    
    chunk_size = max number of characters per chunk
    overlap = number of overlapping characters between chunks
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)