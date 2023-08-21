
import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

@st.cache_resource
def deeplake_embedding(_documents, dataset, token):
    print(f"Going to insert {len(_documents)} to DeepLake")
    # Create the embeddings
    embeddings = OpenAIEmbeddings()
    # Create a DeepLake vectorstore and then load the documents into it
    #db = DeepLake(dataset_path=dataset, embedding_function=embeddings, token=token)
    #db.add_documents(_documents)  
      
    # Create and load the documents in a single action
    db = DeepLake.from_documents(_documents, dataset_path=dataset, embedding=embeddings, overwrite=True)
    print("****** Added to DeepLake vectorstore vectors")
