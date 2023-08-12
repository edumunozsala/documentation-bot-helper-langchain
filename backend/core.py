import os
from typing import Any, List, Dict

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import Pinecone, DeepLake, Chroma

from dotenv import load_dotenv

from constants import INDEX_NAME, VECTORDB, DATABASE_PATH
from utils.pinecone import init_connection


def run_chroma_qa(query: str, vectordir: str, top_k: int) -> Any:
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

def run_pinecone_qa(query: str, index_name: str, top_k: int) -> Any:
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

def run_deeplake_qa(query: str, dataset_path: str, top_k: int) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, read_only=True)
    
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', verbose=True, temperature=0)
    
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":top_k}),
        return_source_documents=True,
    )
    return qa({"query": query})

def run_pinecone_conversational(query: str, index_name: str, top_k: int , chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":top_k}), return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})

def run_deeplake_conversational(query: str, dataset_path: str, top_k: int, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, read_only=True)
    
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":top_k}), return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})

if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()
    