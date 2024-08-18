##
## Try making a vector DB out of Glove text
##
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
import numpy as np

def load_model(init=True):
    # Refresh monster file only as needed 
    if (init):
        with open("glove.6B.300d.txt", 'r', encoding="utf-8") as f:
            vecs = []
            w_list = []
            for line in f:
                values = line.split()
                w_list.append(values[0])
                vecs.append(np.asarray(values[1:], "float32"))
            mat = np.vstack(vecs)
            vectors = normalize(mat, axis=1)
        print("Dimensions: " + str(vectors.shape[1]))
        print("Words: " + str(vectors.shape[0]))
        glove_model = KeyedVectors(vectors.shape[1])
        glove_model.add_vectors(w_list, vectors, replace=True)
        glove_model.save("glove-model")
    else:
        glove_model = KeyedVectors.load("glove-model")
    return glove_model

model = load_model(False)
word = 'carriage'
vector = model[word]   
similar_words = model.most_similar(word, topn=30)
for item in similar_words: 
    print(item)

