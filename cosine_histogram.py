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
    ax1.hist(c, bins=100, density=True)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Similarity of Text Chunks across Novels')
    plt.show()

def main():
    # model_name doesn't matter because no new embedding here 
    cosines1 = []
    tolstoy1 = GetCollection("tolstoy", "chroma3")
    eliot1 = GetCollection("eliot", "chroma3")
    for i in range(eliot1.count()):
        doc = GetDocById(eliot1, i, True)
        vec1 = doc["embeddings"][0]   
        for j in range(tolstoy1.count()):
            doc = GetDocById(tolstoy1, j, True)
            vec2 = doc["embeddings"][0]   
            cosine = np.dot(vec1, vec2)
            cosines1.append(cosine)

    print(f"Mean: {np.mean(cosines1): .2f}")
    print(f"Std Dev: {np.std(cosines1): .2f}")
    print(f"Max: {np.max(cosines1): .2f}")
    print(f"Min: {np.min(cosines1): .2f}")

    cosines2 = []
    tolstoy2 = GetCollection("tolstoy", "chroma2")
    eliot2 = GetCollection("eliot", "chroma2")
    for i in range(eliot2.count()):
        doc = GetDocById(eliot2, i, True)
        vec1 = doc["embeddings"][0]   
        for j in range(tolstoy2.count()):
            doc = GetDocById(tolstoy2, j, True)
            vec2 = doc["embeddings"][0]   
            cosine = np.dot(vec1, vec2)
            cosines2.append(cosine)

    print(f"Mean: {np.mean(cosines2): .2f}")
    print(f"Std Dev: {np.std(cosines2): .2f}")
    print(f"Max: {np.max(cosines2): .2f}")
    print(f"Min: {np.min(cosines2): .2f}")

    fig = plt.figure(figsize=(9,5))
    ax1 = fig.add_subplot(111)
    ax1.hist(cosines2, bins=100, density=True, label= '3-small') # Chroma2
    ax1.hist(cosines1, bins=100, density=True, label = 'ada-002') # Chroma3
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Similarity of Text Chunks across Novels')
    ax1.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()