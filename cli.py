import os
from pathlib import Path
from typing import List, Optional

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings

PERSIST_DIRECTORY = "./chroma_db"
PDF_PATH = "./data/document.pdf"
DEFAULT_MODEL = "gemma:2b"

def load_pdf(pdf_path: str) -> List[dict]:
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []
    
    loader = PyPDFLoader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = loader.load_and_split(text_splitter)
    return [doc.dict() for doc in documents]

def get_llm(model_name: str = DEFAULT_MODEL):
    return Ollama(model=model_name)

def get_vectorstore():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(PERSIST_DIRECTORY):
        if not os.path.exists(PDF_PATH):
            print(f"Error: PDF file not found at {PDF_PATH}")
            return None
            
        documents = load_pdf(PDF_PATH)
        if not documents:
            return None
            
        texts = [doc["page_content"] for doc in documents]
        metadatas = [{"source": doc["metadata"].get("source", "")} for doc in documents]
        
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embedding_function,
            metadatas=metadatas,
            persist_directory=PERSIST_DIRECTORY
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function
        )
    
    return vectorstore

def main():
    print("Initializing LLM and vector store...")
    
    model_name = input(f"Enter model name (default: {DEFAULT_MODEL}): ") or DEFAULT_MODEL
    llm = get_llm(model_name)
    
    vectorstore = get_vectorstore()
    if not vectorstore:
        print("Failed to initialize vector store. Exiting...")
        return
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    print("\nReady! Type 'exit' to quit.")
    print("Enter your question about the document:")
    
    while True:
        try:
            question = input("\nQuestion: ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            print("\nThinking...")
            result = qa_chain({"query": question})
            
            print("\nAnswer:")
            print(result["result"])
            
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"{i}. {doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page', 'N/A')}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
