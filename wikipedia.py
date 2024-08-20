##
## New model, new load procedure
##
from math import dist
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
import numpy as np

def load_model(init=True):
    # Refresh monster file only as needed 
    if (init):
        with open("enwiki_upos_skipgram_300_2_2021.txt", 'r', encoding="utf-8") as f:
            vecs = []
            w_list = []
            for line in f:
                values = line.split()
                token = values[0]
                word, cat = token.split('_')
                if (cat == 'PNOUN') or (cat == 'NUM'):
                    continue
                w_list.append(word)
                vecs.append(np.asarray(values[1:], "float32"))
            mat = np.vstack(vecs)
            vectors = normalize(mat, axis=1)
        print("Dimensions: " + str(vectors.shape[1]))
        print("Words: " + str(vectors.shape[0]))
        model = KeyedVectors(vectors.shape[1])
        model.add_vectors(w_list, vectors, replace=True)
        model.save("wiki-model")
    else:
        model = KeyedVectors.load("wiki-model")
    return model

model = load_model(False)
word = 'gold'
vector = model[word]   
similar_words = model.most_similar(word, topn=30)
for item in similar_words: 
    print(item)









