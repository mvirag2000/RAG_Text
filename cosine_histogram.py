##
## Compare one collection to another  
##
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chroma_utils import GetCollection, CrossDistances

def main():
    model1 = "text-embedding-ada-002"
    tolstoy1 = GetCollection("tolstoy", "chroma3", model1)
    eliot1 = GetCollection("eliot", "chroma3", model1)
    cosines1 = CrossDistances(tolstoy1, eliot1)

    model2 = "text-embedding-3-small"
    tolstoy2 = GetCollection("tolstoy", "chroma2", model2)
    eliot2 = GetCollection("eliot", "chroma2", model2)
    cosines2 = CrossDistances(tolstoy2, eliot2)

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