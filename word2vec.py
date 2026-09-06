##
## Load an embedding and play with word similiarity
## Unfortunately the Google News enbedding is illiterate
##
import gensim.downloader as api
from gensim.models import KeyedVectors
from scipy.stats import differential_entropy
from sklearn.preprocessing import normalize
import re
import numpy as np
import matplotlib.pyplot as plt

def load_model(init=True):
    # Refresh monster file only as needed 
    if (init):
        model = api.load("word2vec-google-news-300")
        words = model.index_to_key
        vecs = []
        w_list = []
        ascii_subset = re.compile(r"^[a-z]+$")
        # Much garbage in this dataset h/t Theo for the cleaning routine
        for w in words:
            if  ascii_subset.match(w): 
                w_list.append(w)
                vecs.append(model[w])
        mat = np.vstack(vecs)
        vectors = normalize(mat, axis=1)
        clean_model = KeyedVectors(vectors.shape[1])
        clean_model.add_vectors(w_list, vectors, replace=True)
        clean_model.save("cleaned-model")
    else:
        clean_model = KeyedVectors.load("cleaned-model")
    return clean_model

def in_tolerance(num1, num2, tol=0.0001):
    if ((num1 - num2) > tol): return False
    if ((num2 - num1) > tol): return False
    return True

def solver():
    giant_score = input("Giant score: ")
    enemy_score = input("Enemy score: ")
    giant = float(giant_score) / 100
    enemy = float(enemy_score) / 100
    tolerance = 0.0001
    for w in keys:
        simil_giant = model.similarity(w, "giant")
        if in_tolerance(simil_giant, giant):
            simil_enemy = model.similarity(w, "enemy")
            if in_tolerance(simil_enemy, enemy):
                print(w)

def similar_words(tol):
    w = input("Input word: ")
    simil_list = []
    for word in words:
        simil = model.similarity(word, w)
        if (simil > tol):
            print(f"{word} {simil:,.6f}")
        simil_list.append(simil)
    plt.hist(simil_list, bins=100)
    plt.title('Distribution of words near \"' + w + '\"')
    plt.xlabel('Similarity')
    plt.ylabel('Words')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.show()

def different_words(tol):
    w = input("Input word: ")
    for word in words:
        simil = model.similarity(word, w)
        if (simil < tol) and (simil > -0.12): # Below -0.15 are junk words
            print(f"{word} {simil:,.6f}")

def distance():
    a = input("Input first word: ")
    b = input("Input second word: ")
    simil = model.similarity(a, b)
    print(simil)

model = load_model(False) # Flag to reinitialize model 
keys = model.key_to_index.keys()
words = list(model.index_to_key)

# solver()
similar_words(0.55)
# distance()
# different_words(-0.05) 
