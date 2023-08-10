import os
import glob
from typing import List

from langchain.docstore.document import Document
from pypdf import PdfReader

import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from utils.deeplake import deeplake_embedding

from constants import INDEX_NAME, VECTORDB_DIR, VECTOR_SIZE, METRIC, VECTORDB, DATABASE_PATH

# Return a list of PDF files in the given directory
def readfiles(path):
    os.chdir(path)
    pdfs = []
    for file in glob.glob("*.pdf"):
       #print(file)
       pdfs.append(os.path.abspath(file))
    
    return pdfs

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
def parse_pdf_files(pdf_files) -> List[str]:
    doc_chunks = []    
    for pdf in pdf_files:
        # Open the file
        print(pdf)
        
        #f = open(pdf, "rb")
        doc = PdfReader(pdf)        
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
            # Create the Text splitter
            # text_splitter = RecursiveCharacterTextSplitter(
            #     chunk_size=1000,
            #     separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            #     chunk_overlap=50,
            #     )
            # Read the chunks of the current page
            chunks = text_splitter.split_text(text)
            
            for j, chunk in enumerate(chunks):
                # Create a document from the chunk and include the metadata
                doc = Document(
                    page_content=chunk, metadata={"source": pdf, "page": i+1, "chunk": j}
                    )
            
                doc_chunks.append(doc)

    return doc_chunks


if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()
    # Load the Arxiv PDF
    pdf="pdf"
    docs= readfiles(pdf)
    print(docs)
    chunks=parse_pdf_files(docs)
    # print(len(chunks))
    # print("\n\n",chunks[50])
    # print("    ",len(chunks[50].page_content))
    # print("\n\n",chunks[80])
    # print("    ",len(chunks[80].page_content))
    # Create the DeepLake vectorstore
    deeplake_embedding(chunks, DATABASE_PATH, os.environ["ACTIVELOOP_TOKEN"])

    