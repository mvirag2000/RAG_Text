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

def GetCollection(name, db: str, model_name):
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

def CreateChromaNative(docs, path, name, model_name):

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
            ids = [str(i + c * chunk_size) for i in range(len(docs))],
        )
        print(f"Saved {len(docs)} chunks to {path}.")

    readme = open(path + "\\readme.txt", 'w')
    readme.write("Created using native Chroma with " + model_name)
    readme.close()

def CreateEmbedding(string, model_name): 
    e = OpenAI().embeddings.create(
        model=model_name,
        input=string,
        encoding_format="float"
    )
    return e.data[0].embedding # Response is an object of which we only want the vector 

def CountTokens(string, chat_model): 
    encoding = tiktoken.encoding_for_model(chat_model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def PrintChunk(chunks, id: int):
    document = chunks[id]
    print(document.page_content)
    print(document.metadata)
    print(CountTokens(document.page_content, "gpt-3.5-turbo"))

def DisplayDocs(title, results): # Results from sililarity search includes DISTANCE
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

def DisplayOne(title, results): # Single DOC from Chroma collection
    window = tk.Tk()
    window.title(title)
    window.geometry("650x300") 
    text_widget = tk.Text(window, wrap="word", font=("Arial", 10))
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(window, orient="vertical", command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    text_widget.config(yscrollcommand=scrollbar.set)
    text_widget.insert(tk.END, "Id: {}".format(results['ids'][0]))
    text_widget.insert(tk.END, str(results['metadatas'][0]) + '\n')
    text_widget.insert(tk.END, results['documents'][0] + '\n\n')
    window.mainloop()

def CrossDistances(collection1, collection2, stats=True):
    cosines = []
    docs1 = collection1.get(include=["embeddings"])
    vectors1 = docs1["embeddings"]
    docs2 = collection2.get(include=["embeddings"])
    vectors2 = docs2["embeddings"]

    vectors1 = np.array(vectors1)
    vectors2 = np.array(vectors2)
    dot_products = np.dot(vectors1, vectors2.T)
    cosines = dot_products.flatten().tolist()

    if stats:
        print(f"Mean: {np.mean(cosines): .2f}")
        print(f"Std Dev: {np.std(cosines): .2f}")
        print(f"Max: {np.max(cosines): .2f}")
        print(f"Min: {np.min(cosines): .2f}")

    return cosines