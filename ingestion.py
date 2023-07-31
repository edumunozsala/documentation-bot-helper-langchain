import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, Chroma, Milvus
from dotenv import load_dotenv

from constants import INDEX_NAME, VECTORDB_DIR, VECTOR_SIZE

import pinecone

def create_pinecone_index(vector_size):
    # First, check if our index already exists. If it doesn't, we create it
    if INDEX_NAME not in pinecone.list_indexes():
    # we create a new index
        print("****** Creating Pinecone index")
        pinecone.create_index(
            name=INDEX_NAME,
            metric='cosine',
            dimension=vector_size  
            )
    print("****** Created Pinecone index")
    
def pinecone_embedding(documents):
    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****** Added to Pinecone vectorstore vectors")

def chroma_embedding(documents, persist_dir):
    print(f"Going to insert {len(documents)} to Chroma")
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embeddings, index_name=INDEX_NAME, persist_directory=persist_dir)
    db.persist()
    print("****** Added to Chroma vectorstore vectors")
    
def milvus_embedding(documents):
    print(f"Going to insert {len(documents)} to Milvus")
    embeddings = OpenAIEmbeddings()
    db = Milvus.from_documents(documents, embeddings, connection_args={"host": "127.0.0.1", "port": "19530"})
    print("****** Added to Milvus vectorstore vectors")
    
    
def ingest_docs() -> None:
    # Set the path to the documentation folder
    doc_path= "langchain-docs\\langchain-docs\\python.langchain.com\\en\\latest\\getting_started"
    # Create a loader with the documentation
    loader = ReadTheDocsLoader(path=doc_path, encoding="ISO-8859-1")
    # Load the documents
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) }documents")
    # Split the documents in chunks of 1000 characters with 100 overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
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
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
    )
    # For persisting purpose the vectors in a directory
    # Not necessary when working with Pinecone
    persist_directory=VECTORDB_DIR
    # Create the Pinecone index
    create_pinecone_index(VECTOR_SIZE)
    # Ingest the documents
    documents = ingest_docs()
    # Embedding the documents
    pinecone_embedding(documents)