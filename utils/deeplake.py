
import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

@st.cache_resource
def deeplake_embedding(_documents, dataset, token):
    print(f"Going to insert {len(_documents)} to DeepLake")
    # Create the embeddings
    embeddings = OpenAIEmbeddings()
    # Create the DeepLake vectorstore
    #db = DeepLake(dataset_path=dataset, embedding_function=embeddings, token=token)
    #db.add_documents(_documents)    
    #DeepLake.from_documents(docs, dataset_path="./my_deeplake/", embedding=embeddings, overwrite=True)
    db = DeepLake.from_documents(_documents, dataset_path=dataset, embedding=embeddings, overwrite=True) #, token=token)
    print("****** Added to DeepLake vectorstore vectors")
    return db