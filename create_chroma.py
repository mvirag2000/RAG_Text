##
## Using native Chroma to create database instead of LangChain 
## ...and native call to OpenAI embeddings  
##
import chunk
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np 
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tiktoken
import shutil

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def create_embedding(string, client):
    e = OpenAI().embeddings.create(
        model="text-embedding-3-small",
        input=string,
        encoding_format="float"
    )
    return e.data[0].embedding # Response is an object of which we only want the vector 

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def parse_book(path):
    loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def display_chunk(chunks, id: int):
    document = chunks[id]
    print(document.page_content)
    print(document.metadata)
    print(num_tokens_from_string(document.page_content))

def create_chroma(chunks, path, name):

    model_name="text-embedding-3-small"
    
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
        )

    print(f"Saved {len(chunks)} chunks to {path}.")
    readme = open(path + "\\readme.txt", 'w')
    readme.write("Created using LangChain and " + model_name)
    readme.close()


def create_chroma_native(docs, path, name):
    # Create vector store without LangChain 

    model_name="text-embedding-3-small"

    # Clear directory or collection
    if os.path.exists(path):
        shutil.rmtree(path)

    client = chromadb.Client(Settings(is_persistent=True, persist_directory=path))
    # if collection_name in [c.name for c in client.list_collections()]: client.delete_collection(name=collection_name)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=model_name)
    collection = client.create_collection(name=name, embedding_function=openai_ef)

    chunk_size = 1000 # Because OpenAI API won't take the whole list 
    chunk_docs = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]

    for c, docs in enumerate(chunk_docs): # This is a nested list: chunk_docs(docs(doc))

        collection.add(documents = [doc.page_content for doc in docs],  
            metadatas = [doc.metadata for doc in docs],
            ids=[str(i + c * chunk_size) for i in range(len(docs))],
        )
        print(f"Saved {len(docs)} chunks to {path}.")

    readme = open(path + "\\readme.txt", 'w')
    readme.write("Created using native Chroma with " + model_name)
    readme.close()

def main():
    collection_name = "tolstoy"
    DATA_PATH = "data/" + collection_name
    CHROMA_PATH = DATA_PATH + "/chroma" # Decided separate DB's were better 

    chunks = parse_book(DATA_PATH)
    display_chunk(chunks, 10)
    create_chroma(chunks, CHROMA_PATH, collection_name) 

if __name__ == "__main__":
    main()
