##
## Pretty much the same routines as word2vec.py only with a better embedding
##
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
from sklearn.decomposition import PCA
import pandas as pd
from wordfreq import top_n_list

vocab = top_n_list('en', 50000)
vocab_index = {word: i for i, word in enumerate(vocab)} 
client = OpenAI()   

def load_vectors(init=True, model="text-embedding-3-small", batch_size=2048):
    # Refresh matrix only as needed 
    if (init):
        all_vectors = []
        for i in range(0, len(vocab), batch_size):
            chunk = vocab[i:i + batch_size]
            resp = client.embeddings.create(model=model, input=chunk)
            all_vectors.extend([d.embedding for d in resp.data])
        matrix = np.array(all_vectors)
        np.save("vocab_embeddings.npy", matrix)
    else:
        matrix = np.load("vocab_embeddings.npy")
    return matrix

def vector(word): # Fresh call to OpenAPI - guaranteed to find a valur
    response = client.embeddings.create(model="text-embedding-3-small", input=word)
    return response.data[0].embedding

def get_vectors(words): # Lookup in matrix - limited vocabulary 
    missing = [w for w in words if w not in vocab_index]
    if missing:
        raise ValueError(f"Not in vocab: {missing}")
    return matrix[[vocab_index[w] for w in words]]

def similarity(a, b):
    a, b = np.array(a), np.array(b)
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

def distance():
    a = input("Input first word: ")
    b = input("Input second word: ")
    simil = similarity(vector(a), vector(b))
    print(simil)

def similar_words(tol):
    w= input("Input word: ")
    target = vector(w)
    sims = matrix @ target   
    mask = sims >= tol
    results = sorted(zip(np.array(vocab)[mask], sims[mask]), key=lambda x: x[1], reverse=True)
    for x in results:
        print(x[0], round(x[1], 5))
    plt.hist(sims, bins=100)
    plt.title('Distribution of words near \"' + w + '\"')
    plt.xlabel('Similarity')
    plt.ylabel('Words')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.show()

def project3D(points, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    three_D = points
    x, y, z = three_D[:,0], three_D[:,1], three_D[:,2]
    ax.scatter(x, y, z)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    for label, x, y, z in zip(labels, x, y, z): 
        ax.text(x, y, z, label)
        print(f"{label} = ({x:.4f}, {y:.4f}, {z:.4f})")
    def rotate(angle): 
        ax.view_init(elev=10, azim=angle)
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2))
    plt.show()
    rot_animation.save('EmbeddingPCA.gif')

matrix = load_vectors(init=False) # Flag to reinitialize matrix 
# similar_words(0.40)
# distance()
value = ['diamond', 'bullion', 'money', 'wealth', 'rand', 'coins', 'bucks']
color = ['yellow', 'grey', 'rose', 'green', 'orange', 'pink', 'blue'] # All within 0.45 of "gold " 
metal = ['iron', 'metal', 'bronze', 'silver', 'chrome', 'palladium']
some_words = ['gold'] + value + color + metal 
print(some_words)
some_vectors = get_vectors(some_words)
pca = PCA(n_components=3)
reduced = pca.fit_transform(some_vectors)
for i, v in enumerate(pca.explained_variance_ratio_, 1):
    print(f"PC{i}: {v:.1%}")
project3D(reduced, some_words)

df = pd.DataFrame(reduced, columns=['PC1', 'PC2', 'PC3'], index=some_words)
print(df.sort_values('PC1'))