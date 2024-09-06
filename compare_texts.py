##
## Read text and vectors from Chroma DB 
##
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chroma_utils import CreateEmbedding, DisplayDocs, GetCollection, GetDocById, BruteScan

def make_chart(c):
    fig = plt.figure(figsize=(9,5))
    ax1 = fig.add_subplot(111)
    ax1.hist(c, bins=50, density=True)
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Probability')
    ax1.set_title('Similarity of Text Chunks across Novels')
    plt.show()

def main():
    tolstoy = GetCollection("tolstoy")
    eliot = GetCollection("eliot")
    for i in range(eliot.count()):
        doc = GetDocById(eliot, i, True)
        query_vec = doc["embeddings"][0]    
        results = tolstoy.query(
            query_embeddings=query_vec,
            n_results = 3)
        if (min(results['distances'][0]) < 0.60): # Distance NOT similarity 
            print(doc["metadatas"][0])
            print(doc["documents"][0])
            print()
            DisplayDocs("Tolstoy Results", results)

if __name__ == "__main__":
    main()