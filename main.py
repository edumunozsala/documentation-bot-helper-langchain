from typing import Set

from backend.core import run_deeplake_conversational,run_pinecone_conversational
import streamlit as st
from streamlit_chat import message

from constants import VECTORDB, DATABASE_PATH, INDEX_NAME

import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
# Clear the prompt input 
def clear_text():
    st.session_state["Prompt"] = ""
    
# Set a tittle for the app
st.header("Documentation Question-Aswering Bot")


prompt = st.text_input("Prompt",  value="", key="Prompt", placeholder="Enter your prompt here..")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response.."):
        # Check the vector database to  use
        if VECTORDB=="DeepLake":
            # Run the LLM on the DeepLake vector database
            generated_response = run_deeplake_conversational(query=prompt, dataset_path=DATABASE_PATH, top_k=2,chat_history=st.session_state["chat_history"])
        elif VECTORDB=="Pinecone":
            # Run the LLM on the Pinecone vector database
            generated_response = run_pinecone_conversational(query=prompt, index_name=INDEX_NAME, top_k=2,chat_history=st.session_state["chat_history"])
        # Read the source document from the metadata in the response
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )
        # Format the response
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))
        # Clean the input prompt
        # clear_text()

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_response)