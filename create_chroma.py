##
## Using native Chroma to create database instead of LangChain 
## ...and native call to OpenAI embeddings  
##
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
DATA_PATH = "data/tolstoy" 
collection_name = "tolstoy"
# DATA_PATH = "data/eliot" 
CHROMA_PATH = DATA_PATH + "/chroma" # Decided separate DB's were better 
ai_client = OpenAI() 

def create_embedding(string, client):
    e = ai_client.embeddings.create(
        model="text-embedding-3-small",
        input=string,
        encoding_format="float"
    )
    return e.data[0].embedding # Response is an object of which we only want the vector 

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)
docs = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(docs)} chunks.")

# Check features of sample chunk, like token size
document = docs[10]
print(document.page_content)
print(document.metadata)
print(num_tokens_from_string(document.page_content))
sample_embedding = create_embedding(document.page_content, ai_client)
print(len(sample_embedding))

# Clear out the database first.
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Delete and re-create COLLECTION 
client = chromadb.Client(Settings(is_persistent=True, persist_directory=CHROMA_PATH))
# if collection_name in [c.name for c in client.list_collections()]: client.delete_collection(name=collection_name)
# openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
collection = client.create_collection(name=collection_name) #, embedding_function=openai_ef)

ids = 0
for doc in docs:
    collection.add(
        metadatas = doc.metadata,
        documents = doc.page_content, 
        ids = str(ids),
        embeddings = create_embedding(doc.page_content, ai_client)
    )
    print(ids)
    ids += 1

print(f"Saved {len(docs)} chunks to {CHROMA_PATH}.")
