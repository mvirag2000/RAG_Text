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
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Probability')
    ax1.set_title('Similarity of Text Chunks across Novels')
    plt.show()

def main():
    cosines = []
    tolstoy = GetCollection("tolstoy")
    eliot = GetCollection("eliot")
    for i in range(eliot.count()):
        doc = GetDocById(eliot, i, True)
        vec1 = doc["embeddings"][0]   
        for j in range(tolstoy.count()):
            doc = GetDocById(tolstoy, j, True)
            vec2 = doc["embeddings"][0]   
            cosine = np.dot(vec1, vec2)
            cosines.append(cosine)

    print(f"Mean: {np.mean(cosines): .2f}")
    print(f"Std Dev: {np.std(cosines): .2f}")
    print(f"Max: {np.max(cosines): .2f}")
    print(f"Min: {np.min(cosines): .2f}")
    make_chart(cosines)

if __name__ == "__main__":
    main()