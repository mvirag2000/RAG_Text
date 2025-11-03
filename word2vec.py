##
## Load an embedding and play with word similiarity
## Unfortunately the Google News enbedding is illiterate
##
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
import re
import numpy as np

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

def in_tolerance(num1, num2, tol=0.00001):
    if ((num1 - num2) > tol): return False
    if ((num2 - num1) > tol): return False
    return True

#giant_score = input("Giant score: ")
#enemy_score = input("Enemy score: ")
model = load_model(False)
keys = model.key_to_index.keys()
words = list(model.index_to_key)

# The Marinade Digression
for word in words:
    simil_marinade = model.similarity(word, "marinade")
    if (simil_marinade > 0.70):
        print(f"{word} {simil_marinade:,.6f}")

stop

# Semantle Solver
giant = float(giant_score) / 100
enemy = float(enemy_score) / 100
tolerance = 0.0001
for w in keys:
    simil_giant = model.similarity(w, "giant")
    if in_tolerance(simil_giant, giant):
        simil_enemy = model.similarity(w, "enemy")
        if in_tolerance(simil_enemy, enemy):
            print(w)





