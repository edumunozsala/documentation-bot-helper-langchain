import os
from typing import Any

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone, Chroma

from dotenv import load_dotenv

from constants import INDEX_NAME
from utils.pinecone import init_connection


def run_chroma_llm(query: str, vectordir, top_k) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma(persist_directory=vectordir, embedding_function=embeddings)
    
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":top_k}),
        return_source_documents=True,
    )
    return qa({"query": query})

def run_pinecone_qa(query: str, index_name, top_k) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', verbose=True, temperature=0)
    
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":top_k}),
        return_source_documents=True,
    )
    return qa({"query": query})


if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()
    # Connect to Pinecone
    init_connection(os.environ["PINECONE_API_KEY"], os.environ["PINECONE_ENVIRONMENT_REGION"])
    # Run the QA chain
    results=run_pinecone_qa(query="What is Cross-Validation?", index_name=INDEX_NAME, top_k=2)
    print(" ANSWER: ", results['result'])
    print(" SOURCE: ", results['source_documents'])
    