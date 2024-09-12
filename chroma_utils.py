##
## Native Chroma utilities for OpenAI (without LangChain)
##
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import numpy as np
import shutil
import os
import openai
from openai import OpenAI
import tiktoken
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
model_name = 'text-embedding-3-small' 

def GetCollection(name, db: str):
    chroma_path = "data/" + name + '/' + db
    client = chromadb.Client(Settings(is_persistent=True, persist_directory=chroma_path))
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=model_name)
    # print([c.name for c in client.list_collections()])
    collection = client.get_collection(name=name, embedding_function=openai_ef)
    return collection

def GetDocById(collection, id: int, include=False):
    doc_id = str(id)
    if include:
        result = collection.get(
        ids=[doc_id],
        include=["embeddings", "metadatas", "documents"], # Embeddings are excluded unless you ask for them 
        )
    else:
        result = collection.get(
        ids=[doc_id],
        )
    return result

def BruteScan(my_vec, collection):
    dist_list = []
    for i in range(collection.count()):
        doc = GetDocById(collection, i, True)
        vec = doc["embeddings"][0]    
        cosine = np.dot(my_vec, vec)
        dist_list.append(cosine)
    return dist_list

def CreateChromaNative(docs, path, name):

    # Clear directory or collection
    if os.path.exists(path):
        shutil.rmtree(path)

    client = chromadb.Client(Settings(is_persistent=True, persist_directory=path))
    # if collection_name in [c.name for c in client.list_collections()]: client.delete_collection(name=collection_name)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=model_name)
    collection = client.create_collection(name=name, 
        metadata={"hnsw:space": "cosine"}, # Actually, 1-Dot so cosine "distance" not "similarity" 
        embedding_function=openai_ef)

    chunk_size = 100 # Because OpenAI API won't take the whole list 
    chunk_docs = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]

    for c, docs in enumerate(chunk_docs): # This is a nested list: chunk_docs(docs(doc))

        collection.add(documents = [doc.page_content for doc in docs], # This line throws error 400 if docs is too big (length or chunk size)
            metadatas = [doc.metadata for doc in docs],
            ids=[str(i + c * chunk_size) for i in range(len(docs))],
        )
        print(f"Saved {len(docs)} chunks to {path}.")

    readme = open(path + "\\readme.txt", 'w')
    readme.write("Created using native Chroma with " + model_name)
    readme.close()

def CreateEmbedding(string): 
    e = OpenAI().embeddings.create(
        model=model_name,
        input=string,
        encoding_format="float"
    )
    return e.data[0].embedding # Response is an object of which we only want the vector 

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def DisplayChunk(chunks, id: int):
    document = chunks[id]
    print(document.page_content)
    print(document.metadata)
    print(num_tokens_from_string(document.page_content))

def DisplayDocs(title, results):
    window = tk.Tk()
    window.title(title)
    window.geometry("650x300") 
    text_widget = tk.Text(window, wrap="word", font=("Arial", 10))
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(window, orient="vertical", command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    text_widget.config(yscrollcommand=scrollbar.set)
    length = len(results['ids'][0])
    for idx in range(length):
        text_widget.insert(tk.END, "Id: {}".format(results['ids'][0][idx]))
        text_widget.insert(tk.END, " Distance:{:.2f}\n".format(results['distances'][0][idx]))
        text_widget.insert(tk.END, str(results['metadatas'][0][idx]) + '\n')
        text_widget.insert(tk.END, results['documents'][0][idx] + '\n\n')
    window.mainloop()