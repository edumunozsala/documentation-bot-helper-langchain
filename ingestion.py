import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

#from consts import INDEX_NAME

#pinecone.init(
#    api_key=os.environ["PINECONE_API_KEY"],
#    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
#)

def pinecone_embedding():
    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents[3969:], embeddings, index_name=INDEX_NAME)
    print("****** Added to Pinecone vectorstore vectors")


def chroma_embedding():
    print(f"Going to insert {len(documents)} to Chroma")
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents[3969:], embeddings, index_name=INDEX_NAME)
    #print("****** Added to Pinecone vectorstore vectors")
    pass
    
def ingest_docs() -> None:
    # Set the path to the documentation folder
    doc_path= "langchain-docs\\langchain-docs\\python.langchain.com\\en\\latest"
    # Create a loader with the documentation
    loader = ReadTheDocsLoader(path=doc_path, encoding="ISO-8859-1")
    # Load the documents
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) }documents")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    #print(f"Going to insert {len(documents)} to Pinecone")
    #embeddings = OpenAIEmbeddings()
    #Pinecone.from_documents(documents[3969:], embeddings, index_name=INDEX_NAME)
    #print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()