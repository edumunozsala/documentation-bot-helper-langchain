import os

from dotenv import load_dotenv

from constants import INDEX_NAME, VECTORDB, DATABASE_PATH
from utils.pinecone import init_connection
from backend.core import run_deeplake_conversational, run_pinecone_conversational


if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()
    # Select the vector store    
    if VECTORDB=="Pinecone":
        # Connect to Pinecone
        init_connection(os.environ["PINECONE_API_KEY"], os.environ["PINECONE_ENVIRONMENT_REGION"])
        # Run the QA chain
        results=run_pinecone_conversational(query="What is Cross-Validation?", index_name=INDEX_NAME, top_k=2)
        print(" ANSWER: ", results['result'])
        print(" SOURCE: ", results['source_documents'])
    elif VECTORDB=="DeepLake":
        # Run the QA chain
        query="What is Cross-Validation?"
        results=run_deeplake_conversational(query=query, dataset_path=DATABASE_PATH, top_k=2)
        #print(" ANSWER: ", results['result'])
        #print(" SOURCE: ", results['source_documents'])
        print(results['answer'])
        chat_history = [(query, results["answer"])]
        results=run_deeplake_conversational(query="Which variant is most commonly used?", dataset_path=DATABASE_PATH, top_k=2, chat_history=chat_history)
        print(results['answer'])
        