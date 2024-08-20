##
## Try making a vector DB out of Glove text
##
from math import dist
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
word = 'sigarms'
vector = model[word]   
similar_words = model.most_similar(word, topn=50)
for item in similar_words: 
    print(item)

word1 = 'hot'
word2 = 'cold'
similarity = model.similarity(word1, word2)
#print(f'Similarity: {similarity}')

def cosine(word1, word2):
    vector1 = model[word1]
    vector2 = model[word2]
    dot = np.dot(vector1, vector2)
    # return similarity # In case vectors aren't normed 
    return dot
    norm1 = np.linalg.norm(vector1, ord=2)
    norm2 = np.linalg.norm(vector2, ord=2)
    similarity = dot / (norm1 * norm2)
    return similarity

def euclid(word1, word2):
    vector1 = model[word1]
    vector2 = model[word2]
    dist = np.linalg.norm(vector1 - vector2)
    return dist

distance = euclid(word1, word2)
print(f'Distance: {distance}')
distance = np.sqrt(2 * (1 - similarity)) # Same as Euclid if vectors are normed 
print(f'Distance: {distance}')
distance = model.distance(word1, word2) # This is WMD distance not Euclid 
print(f'Distance: {distance}')





