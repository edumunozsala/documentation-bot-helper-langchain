from typing import Set, List

import streamlit as st
from streamlit_chat import message
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from utils.deeplake import deeplake_embedding
from backend.core import run_deeplake_conversational, run_pinecone_conversational
from constants import DATABASE_PATH, ACCEPT_MULTIPLE_FILES, VECTORDB, INDEX_NAME

import os
import re
from dotenv import load_dotenv


# Clean a PDF page
def clean_page(page: str) -> str:
    # Merge hyphenated words
    page = re.sub(r"(\w+)-\n(\w+)", r"\1\2", page)
    # Fix newlines in the middle of sentences
    page = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", page.strip())
    # Remove multiple newlines
    page = re.sub(r"\n\s*\n", "\n\n", page)
    # Remove multiple spaces
    page = re.sub(r"\s+", " ", page)
    
    return page

# Read and return the cleaned pages of the given PDF files
@st.cache_data
def parse_pdf_files(pdf_files) -> List[str]:
        doc_chunks = []    
    #for pdf in pdf_files:
        # Read the PDF file
        doc = PdfReader(pdf_files)        
        print(doc.pages)

        # Create the Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=50,
                )
        
        for i, page in enumerate(doc.pages):
            # Clean the page
            text = clean_page(page.extract_text())
            # Read the chunks of the current page
            chunks = text_splitter.split_text(text)
            # Iterate over the chunks
            for j, chunk in enumerate(chunks):
                # Create a document from the chunk and include the metadata
                doc = Document(
                    page_content=chunk, metadata={"source": pdf_files.name, "page": i+1, "chunk": j}
                    )
            
                doc_chunks.append(doc)

        return doc_chunks

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = ""
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

# Load the environment variables
load_dotenv()
# Clear the prompt input 
# def clear_text():
#     st.session_state["Prompt"] = ""
    
# Set a tittle for the app
st.title("Question-Answering Chatbot with Memory 🧠 ")
st.header("Interact with your PDF documents 📜 using a Question-Answering Chatbot 🤖")
st.subheader('Load a PDF document and ask it questions')

# Set up the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Upload PDF File
    2. Enter Your Secret Key for Embeddings
    3. Perform Q&A

    **Note : File content and API key not stored in any form.**
    """
)

# Upload PDF file
uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"], accept_multiple_files= ACCEPT_MULTIPLE_FILES)

if uploaded_file:
    # Check if uploaded_fle is a string
    # if not ACCEPT_MULTIPLE_FILES:
    #     # Convert the string to a list
    #     uploaded_file = [uploaded_file]
    # # Check if the uploaded_file is an empty list
    # if not uploaded_file:
    #     st.error("Please upload a PDF file")
    #     st.stop()
        
    # Parse the uploaded PDF files
    chunks = parse_pdf_files(uploaded_file)
    # Create the DeepLake vectorstore
    deeplake_embedding(chunks, DATABASE_PATH, os.environ["ACTIVELOOP_TOKEN"])
        
    # Ask for t paper to chat with
    prompt = st.text_input("Prompt",  value="", placeholder="Enter your prompt here..")
    # # Set the paper in the session state 
    # if "user_paper" not in st.session_state:
    #     st.session_state["user_paper"] = []

    # if paper:
        
    # prompt = st.text_input("Prompt",  value="", key="Prompt", placeholder="Enter your prompt here..")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []

    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


    # def create_sources_string(source_urls: Set[str]) -> str:
    #     if not source_urls:
    #         return ""
    #     sources_list = list(source_urls)
    #     sources_list.sort()
    #     sources_string = ""
    #     for i, source in enumerate(sources_list):
    #         sources_string += f"{i+1}. {source}\n"
    #     return sources_string


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
                #f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
                f"{generated_response['answer']} \n\n"
            )

            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append((prompt, generated_response["answer"]))
    #         # Clean the input prompt
    #         # clear_text()

        if st.session_state["chat_answers_history"]:
            for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
                message(user_query, is_user=True, key="query")
                message(generated_response,key="response")

                # With a streamlit expander  
                with st.expander('Sources:'):
                    # Write out the relevant pages
                    st.write(create_sources_string(sources))

                # With a streamlit expander  
                with st.expander('Memory:'):
                    # Write out the relevant pages
                    st.write(st.session_state["chat_history"])
