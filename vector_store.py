import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(data_dir="data", index_path="faiss_index"):
    print(f"Loading documents from {data_dir}...")
    documents = []
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found.")
        return None
        
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                text = f.read()
                # Use filename as metadata for source tracking
                documents.append(Document(page_content=text, metadata={"source": file}))
                
    if not documents:
        print("No documents found to embed.")
        return None

    print(f"Loaded {len(documents)} documents. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks. Generating FAISS embeddings with Ollama...")
    # Using llama3:8b as it was available in user's env previously.
    embeddings = OllamaEmbeddings(model="llama3:8b")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    
    print(f"SUCCESS: FAISS vector store saved to {index_path}")
    return vector_store

def get_vector_store(index_path="faiss_index"):
    """Loads the existing FAISS index from disk."""
    embeddings = OllamaEmbeddings(model="llama3:8b")
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return None

if __name__ == "__main__":
    build_vector_store()
