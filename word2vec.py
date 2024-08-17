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
word = 'apple'
vector = model[word]   
similar_words = model.most_similar(word, topn=30)
for item in similar_words: 
    print(item)

