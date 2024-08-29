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

def main():

    tolstoy = set_paths("tolstoy")
    eliot = set_paths("eliot")

    #for i in range(eliot.count()):
    #    r = get_by_id(eliot, i)
    #    print(r["ids"][0] + " " + r["documents"][0][:40] + " " + r["metadatas"][0]["source"] + " " + str(r["metadatas"][0]["start_index"]))

    cosines = []
    for a in range(tolstoy.count()):
        a_doc = get_by_id(tolstoy, a, True)
        a_vec = a_doc["embeddings"][0]    
        for b in range(eliot.count()):
            b_doc = get_by_id(eliot, b, True)
            b_vec = b_doc["embeddings"][0]    
            cosine = np.dot(a_vec, b_vec)
            #if (cosine > 0.82):
            #    print(a_doc["ids"][0] + " " + a_doc["documents"][0][:40] + " " + a_doc["metadatas"][0]["source"] + " " + str(a_doc["metadatas"][0]["start_index"]))
            #    print(b_doc["ids"][0] + " " + b_doc["documents"][0][:40] + " " + b_doc["metadatas"][0]["source"] + " " + str(b_doc["metadatas"][0]["start_index"]))
            #    print()
            cosines.append(cosine)
      
    fig = plt.figure(figsize=(9,5))
    ax1 = fig.add_subplot(111)
    ax1.hist(cosines, bins=50, density=True)
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Probability')
    ax1.set_title('Similarity of Text Chunks across Novels')
    plt.show()

if __name__ == "__main__":
    main()