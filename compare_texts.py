##
## Read text and vectors from Chroma DB 
##
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

def set_paths(name: str):
    chroma_path = "data/" + name + "/chroma"
    client = chromadb.Client(Settings(is_persistent=True, persist_directory=chroma_path))
    # print([c.name for c in client.list_collections()])
    collection = client.get_collection(name=name)
    # db = Chroma(persist_directory=chroma_path)
    return collection

def get_by_id(collection, id: int, include=False):
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

def brute_scan(my_vec, collection):
    dist_list = []
    for i in range(collection.count()):
        doc = get_by_id(collection, i, True)
        vec = doc["embeddings"][0]    
        cosine = np.dot(my_vec, vec)
        dist_list.append(cosine)
    return dist_list

def make_chart(c):
    fig = plt.figure(figsize=(9,5))
    ax1 = fig.add_subplot(111)
    ax1.hist(c, bins=50, density=True)
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Probability')
    ax1.set_title('Similarity of Text Chunks across Novels')
    plt.show()

def main():
    tolstoy = set_paths("tolstoy")
    eliot = set_paths("eliot")
    cosines = []
    for i in range(tolstoy.count()):
        doc = get_by_id(tolstoy, i, True)
        vec = doc["embeddings"][0]    
        dist_list = brute_scan(vec, eliot)
        cosines.extend(dist_list)
    print(f"Mean: {np.mean(cosines): .2f}")
    print(f"Std Dev: {np.std(cosines): .2f}")
    print(f"Max: {np.max(cosines): .2f}")
    make_chart(cosines) 

if __name__ == "__main__":
    main()