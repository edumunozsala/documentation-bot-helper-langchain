
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from langchain.vectorstores import DeepLake

def deeplake_embedding(documents, dataset, token):
    print(f"Going to insert {len(documents)} to DeepLake")
    # Create the embeddings
    embeddings = OpenAIEmbeddings()
    # Create the DeepLake vectorstore
    db = DeepLake(dataset_path=dataset, embedding_function=embeddings, token=token)
    db.add_documents(documents)    
    print("****** Added to DeepLake vectorstore vectors")