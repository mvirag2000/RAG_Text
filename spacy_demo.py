import spacy
import numpy as np
from sense2vec import Sense2Vec

s2v = Sense2Vec().from_disk("C:/Users/Mark/Desktop/s2v_reddit_2015_md/s2v_old")
query = "gold|ADJ"
# assert query in s2v
vector = s2v[query]
freq = s2v.get_freq(query)
most_similar = s2v.most_similar(query, n=5) 

print(freq)
print(vector.shape)
print(np.linalg.norm(vector, ord=2))
print(most_similar)

