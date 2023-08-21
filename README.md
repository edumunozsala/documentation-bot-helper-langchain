# Develop a LLM powered Chatbot to chat with a PDF file using Langchain and Vector DB
## Description 

We want to extract information from private data in the form of a PDF file. A chatbot interface is a user friendly approach and we need to apply a memory mechanism to keep track of the conversation and optimize how we interact with the Chatbot. To handle the amount of information that a pdf file can contain, we load all that information in a vector database, including the embedding vectors that compress that information.

Every time a new question is inserted, we call an embedding model to transform it to a vector and, after that, we send the query to the vector db engine and using a semantic search the top k most similar pieces of text are returned. All that context is sent to the LLM, ChatGPT, to get an accurate response.
### Chatbot App
Streamlit is a very helpful tool to build a simple demo app for many machine learning tasks. It is a simple app to show how this model work. Robustness and eficiency is not the goal of this app.

We have uploaded the app to Streamlit Community Cloud to share it with the community.

## Content

The source code provides method and functions to work with Pinecone and Deeplake vector databases, but the final app is built on top of Deeplake.

In future realeses we would try to select the database to use and other configuration options.

- main.py: The Streamlit app
- utils: source code to handle vector databases
- constant.py: Dome config parameters for the database connection.
- .py files: there is a .py file for every stage of the process: ingestion, retrival, conversational and chat. 

## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

These notebooks are under a Apache 2.0 license.

