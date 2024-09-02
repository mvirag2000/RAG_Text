##
## Read text and vectors from Chroma DB 
##
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chroma_utils import GetCollection, GetDocById, BruteScan

load_dotenv()

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
    cosines = []
    for i in range(2):
        doc = GetDocById(tolstoy, i, True)
        vec = doc["embeddings"][0]    
        dist_list = BruteScan(vec, eliot)
        cosines.extend(dist_list)
    print(f"Mean: {np.mean(cosines): .2f}")
    print(f"Std Dev: {np.std(cosines): .2f}")
    print(f"Max: {np.max(cosines): .2f}")
    make_chart(cosines) 

if __name__ == "__main__":
    main()