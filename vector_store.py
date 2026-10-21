"""
Module for managing document indexing and vector-based retrieval.
Uses FAISS and Ollama embeddings for memory management.
"""
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def build_vector_store(data_dir="data", index_path="faiss_index"):
    """
    Reads text documents from a directory, splits them into chunks, and creates 
    a FAISS vector index saved to disk.
    
    Args:
        data_dir (str): Directory containing the source .txt credit memos.
        index_path (str): The local path where the FAISS index files will be saved.
        
    Returns:
        FAISS: The initialized vector store object, or None if no documents were found.
    """
    if not os.path.exists(data_dir):
        return None
        
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(Document(page_content=text, metadata={"source": file}))
                
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    llama_model = os.getenv("LLAMA_MODEL", "gemma2:tiny")
    embeddings = OllamaEmbeddings(model=llama_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    
    print(f"SUCCESS: FAISS vector store saved to {index_path} using model {llama_model}")
    return vector_store

def get_vector_store(index_path="faiss_index"):
    """
    Loads an existing FAISS index from the local file system.
    
    Args:
        index_path (str): The folder path where the FAISS index is located.
        
    Returns:
        FAISS: The loaded vector store object, or None if it doesn't exist.
    """
    llama_model = os.getenv("LLAMA_MODEL", "gemma2:tiny")
    embeddings = OllamaEmbeddings(model=llama_model)
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return None

if __name__ == "__main__":
    build_vector_store()
