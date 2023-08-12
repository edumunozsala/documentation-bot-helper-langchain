import pinecone

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

def init_connection(api_key, environment):
    # Initialize the Pinecone connection    
    pinecone.init(
        api_key=api_key,
        environment=environment,
    )

def create_pinecone_index(vector_size, metric, index_name):
    # First, check if our index already exists. If it doesn't, we create it
    if index_name not in pinecone.list_indexes():
    # we create a new index
        print("****** Creating Pinecone index")
        pinecone.create_index(
            name=index_name,
            metric=metric,
            dimension=vector_size  
            )
    print("****** Created Pinecone index")

def pinecone_embedding(documents, index_name):
    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name=index_name)
    print("****** Added to Pinecone vectorstore vectors")
