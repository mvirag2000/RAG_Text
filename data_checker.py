##
## Read text and vectors from Chroma DB 
##
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chroma_utils import CrossDistances, DisplayOne, GetCollection, GetDocById
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import umap
import hdbscan 
sns.set_palette(sns.color_palette() )

collection_name = "thackeray"
collection_path = "chroma4" 
model_name = "text-embedding-3-large"

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def main():

    print("Collection: " + collection_name)

    collection = GetCollection(collection_name, collection_path, model_name)
    count = collection.count()
    print(f"Documents: {count:,}")

    docs = collection.get(include=["embeddings", "documents"])
    vectors = docs["embeddings"]
    dims = len(vectors[0])
    print(f"Dimensions: {dims:,}")
    chunk = len(docs["documents"][0])
    print(f"Chunk size: {chunk:,}")
    
    rng = np.random.default_rng()
    rand_item = rng.integers(0, count)
    DisplayOne("Some doc", GetDocById(collection, rand_item))
    
    n_comps = 20
    pca = PCA(n_components=n_comps)
    reduced = pca.fit_transform(vectors)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum(): 0.2f} by: {n_comps:,} components")

    fig1 = plt.figure(figsize=(11,5))
    ax = fig1.add_subplot(121)
    ticks = np.arange(pca.n_components_)+1
    ax.plot(ticks,
        pca.explained_variance_ratio_.cumsum(),
        marker='o')
    ax.set_xlabel('Principal Component');
    ax.set_ylabel('Cumulative Variance')
    ax.set_title('Proportion of Variance Explained')
    ax.set_ylim([0,1])
    ax.set_xticks(ticks)

    self_similarity = CrossDistances(collection, collection, stats=False)
    max_dist = 1 - np.min(self_similarity)
    print(f"Max distance: {max_dist: .2f}")
    self_similarity = [x for x in self_similarity if x < 0.99] # Remove the same-doc results 

    ax1 = fig1.add_subplot(122)
    ax1.hist(self_similarity, bins=200, density=True)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Similarity among Docs of Collection')
    plt.show()

    linkage = 'complete'
    cluster_model = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage=linkage, compute_distances=True, distance_threshold=0)
    clusters = cluster_model.fit_predict(vectors)
 
    plt.title("Hierarchical Clustering: " + collection_name + ", Linkage: " + linkage)
    plot_dendrogram(cluster_model, truncate_mode="level", p=4, color_threshold=(max_dist*0.75))
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    neighbors = int(count * 0.10)
    manifold = umap.UMAP(
        n_neighbors=neighbors,
        metric='cosine',
        min_dist=0.0,
        n_components=3,
    ).fit_transform(vectors)

    # This approach flatters the reduction routine
    # By clustering only in the reduced 3D space
    # A better test would be to do clusters in high-D space 
    # Then see if the reduction preserves them

    size = int(count * 0.02)
    hdb_model = hdbscan.HDBSCAN(
        min_cluster_size=size,
        metric='euclidean' # Manifold is simple 3D 
    ).fit(manifold)
    labels = hdb_model.labels_
    print(np.max(labels)) # Values < 0 are noise 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        manifold[:, 0],
        manifold[:, 1],
        manifold[:, 2],
        c=labels,
        s=7,
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title("UMAP projection of " + collection_name)
    plt.show()

if __name__ == "__main__":
    main()