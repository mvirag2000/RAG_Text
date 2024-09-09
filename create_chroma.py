##
## Using native Chroma or LangChain to create database
##
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np 
import shutil
from chroma_utils import CreateChromaNative, DisplayChunk, DisplayDocs

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
model_name="text-embedding-ada-002"

def create_chroma(chunks, path, name):
        
    # Clear out the database first.
    if os.path.exists(path):
        shutil.rmtree(path)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, 
        OpenAIEmbeddings(model=model_name), 
        persist_directory=path, 
        ids=[str(i) for i in range(len(chunks))],
        collection_name=name,
        collection_metadata={"hnsw:space": "cosine"},
        )
    print(f"Saved {len(chunks)} chunks to {path}.")
    readme = open(path + "\\readme.txt", 'w')
    readme.write("Created using LangChain and " + model_name)
    readme.close()

def parse_book(path):
    loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    '''
    text_splitter = SemanticChunker(
        OpenAIEmbeddings(model=model_name), 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85 # Higher number means fewer breaks -> larger chunks 
    )
    '''
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def main():
    collection_name = "eliot"
    DATA_PATH = "data/" + collection_name
    CHROMA_PATH = DATA_PATH + "/chroma3" # Decided separate DB's were better 

    chunks = parse_book(DATA_PATH)
    DisplayChunk(chunks, 10)
    create_chroma(chunks, CHROMA_PATH, collection_name) 

if __name__ == "__main__":
    main()
