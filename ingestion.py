import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from constants import INDEX_NAME, VECTORDB_DIR, VECTOR_SIZE, METRIC
from utils.pinecone import init_connection, create_pinecone_index, pinecone_embedding

   
def load_docs_from_path(doc_path, encoding="ISO-8859-1"):
    # Create a loader with the documentation
    loader = ReadTheDocsLoader(path=doc_path, encoding=encoding)
    # Load the documents
    raw_documents = loader.load()
    # Show count 
    print(f"loaded {len(raw_documents) }documents")
    return raw_documents
    
def load_docs_from_pdfs(doc_path):
    #Check if doc_path is a directory
    if os.path.isdir(doc_path):
        # Create a loader with the documentation
        loader = PyPDFDirectoryLoader(path=doc_path)
        # Load the documents
    else:
        # Create a loader with the documentation
        loader = PyPDFLoader(path=doc_path)
        
    # Load the documents
    raw_documents = loader.load()
    # Show count 
    print(f"loaded {len(raw_documents) }documents")
    return raw_documents
    
def ingest_docs(raw_documents) -> None:
    # Split the documents in chunks of 1000 characters with 100 overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")
    # Update the metadata with the source url
    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    return documents
    

if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()
    # Initialize the Pinecone connection    
    init_connection(os.environ["PINECONE_API_KEY"], os.environ["PINECONE_ENVIRONMENT_REGION"])
    # For persisting purpose the vectors in a directory
    # Not necessary when working with Pinecone
    persist_directory=VECTORDB_DIR
    # Create the Pinecone index
    create_pinecone_index(VECTOR_SIZE, metric=METRIC, index_name=INDEX_NAME)
    # Set the path to the documentation folder
    #doc_path= "langchain-docs\\langchain-docs\\python.langchain.com\\en\\latest\\getting_started"
    doc_path= "pdf"
    # Read the docs
    #raw_documents = load_docs_from_path(doc_path, encoding="ISO-8859-1")
    # Read the docs from PDFs
    raw_documents = load_docs_from_pdfs(doc_path)
    # Ingest the documents
    documents = ingest_docs(raw_documents)
    # Embedding the documents
    pinecone_embedding(documents, INDEX_NAME)