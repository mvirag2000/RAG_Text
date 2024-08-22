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
        # Much garbage in this dataset h/t Theo for cleaning routine
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

model = load_model(False)
keys = model.key_to_index.keys()
giant = -0.0084
enemy = 0.0835
tolerance = 0.0001
for w in keys:
    simil_giant = model.similarity(w, "giant")
    if ((simil_giant < giant + tolerance) and  (simil_giant > giant - tolerance)):
        simil_enemy = model.similarity(w, "enemy")
        if ((simil_enemy < enemy + tolerance) and  (simil_enemy > enemy - tolerance)):
            print(w)

#word = 'apple'
#vector = model[word]   
#similar_words = model.most_similar(word, topn=5)
#for item in similar_words: 
#    print(item)

