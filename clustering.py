##
## Find context clusters in database  
##
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from chroma_utils import GetCollection, GetDocById
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans 
from sklearn.metrics import silhouette_samples, silhouette_score
import umap

collection_name = "tolstoy"
collection_path = "chroma3" 
model_name = "text-embedding-ada-002"

def main():

    print("Collection: " + collection_name)

    collection = GetCollection(collection_name, collection_path, model_name)
    count = collection.count()
    print(f"Documents: {count:,}")

    docs = collection.get(include=["embeddings"])
    vectors = docs["embeddings"]
    dims = len(vectors[0])
    print(f"Dimensions: {dims:,}")

    # First find a good clustering model
    # Then see if UMAP can visualize it
    '''
    size = int(count*0.02)
    model = HDBSCAN(
        min_cluster_size=5,
        # metric='cosine',  # Doesn't matter with normed vectors 
        store_centers='medoid',
        copy=True,
    ).fit(vectors)

    model = AgglomerativeClustering(
        n_clusters=2, 
        # metric='cosine', 
        linkage='complete', 
        compute_distances=True, 
        distance_threshold=None,
    ).fit(vectors)
    '''
    # Oddly, K-Means seems to agree best with UMAP 
    size = 4
    model = KMeans(
        n_clusters=size,
        copy_x = True
    ).fit(vectors)

    labels = model.labels_
    unique, counts = np.unique(labels, return_counts=True)
    label_names = dict(zip(unique, counts))
    print(label_names)

    neighbors = int(count * 0.10)
    manifold = umap.UMAP(
        # n_neighbors=neighbors,
        # metric='cosine',
        min_dist=0.0,
        n_components=3,
    ).fit_transform(vectors)

    fig = plt.figure()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))
    cmap = mcolors.ListedColormap(colors)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        manifold[:, 0],
        manifold[:, 1],
        manifold[:, 2],
        c=labels,
        cmap=cmap,
        s=7,
    )
    label_text = [f'Class {i}' for i in unique]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) 
           for i in range(len(unique))]
    plt.legend(handles, label_text)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title("UMAP projection of " + collection_name)
    plt.show()

if __name__ == "__main__":
    main()