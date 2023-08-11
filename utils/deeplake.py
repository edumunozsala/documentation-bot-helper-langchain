
import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from langchain.vectorstores import DeepLake

@st.cache_data
def deeplake_embedding(_documents, dataset, token):
    print(f"Going to insert {len(_documents)} to DeepLake")
    # Create the embeddings
    embeddings = OpenAIEmbeddings()
    # Create the DeepLake vectorstore
    db = DeepLake(dataset_path=dataset, embedding_function=embeddings, token=token)
    db.add_documents(_documents)    
    print("****** Added to DeepLake vectorstore vectors")