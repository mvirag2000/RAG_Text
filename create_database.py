##
## Starter RAG project courtesy of Pixegami 
##
from langchain_community.document_loaders import DirectoryLoader
# from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
DATA_PATH = "data/tolstoy" 
# DATA_PATH = "data/eliot" 
CHROMA_PATH = DATA_PATH + "/chroma" 

loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)
chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

document = chunks[10] # Display arbitrary chunk to look at 
print(document.page_content)
print(document.metadata)

# Clear out the database first.
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Create a new DB from the documents.
db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
db.persist()
print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
