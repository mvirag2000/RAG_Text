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
from chroma_utils import CreateChromaNative, PrintChunk

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

collection_name = "tolstoy"
collection_path = "chroma5" 
model_name = "text-embedding-3-large"

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
        chunk_size=800,
        chunk_overlap=60,
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

    chunks = parse_book("data/" + collection_name)
    PrintChunk(chunks, 10)
    create_chroma(chunks, "data/" + collection_name + "/" + collection_path, collection_name) 

if __name__ == "__main__":
    main()
