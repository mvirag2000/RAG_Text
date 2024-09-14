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

collection_name = "thackeray"
collection_path = "chroma3" 
model_name = "text-embedding-3-small"

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

    doc = GetDocById(collection, 0, True)
    dims = len(doc["embeddings"][0])
    print(f"Dimensions: {dims:,}")

    docs = collection.get(include=["embeddings"])
    vectors = docs["embeddings"]
    ids = docs["ids"]
    
    rng = np.random.default_rng()
    rand_item = rng.integers(0, count)
    DisplayOne("Some doc", GetDocById(collection, rand_item))
    
    n_comps = 60
    pca = PCA(n_components=n_comps)
    reduced = pca.fit_transform(vectors)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum(): 0.2f} by: {n_comps:,} components")

    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    ticks = np.arange(pca.n_components_)+1
    ax.plot(ticks,
        pca.explained_variance_ratio_.cumsum(),
        marker='o')
    ax.set_xlabel('Principal Component');
    ax.set_ylabel('Cumulative Proportion of Variance Explained')
    ax.set_ylim([0,1])
    ax.set_xticks(ticks)
    plt.show()

    self_similarity = CrossDistances(collection, collection, stats=False)
    max_dist = 1 - np.min(self_similarity)
    print(f"Max distance: {max_dist: .2f}")
    self_similarity = [x for x in self_similarity if x < 0.99] # Remove the same-doc results 

    fig = plt.figure(figsize=(9,5))
    ax1 = fig.add_subplot(111)
    ax1.hist(self_similarity, bins=200, density=True)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Similarity among Docs of Collection')
    plt.show()

    linkage = 'complete'
    cluster_model = AgglomerativeClustering(n_clusters=6, metric='cosine', linkage=linkage, compute_distances=True)
    clusters = cluster_model.fit_predict(vectors)
 
    plt.title("Hierarchical Clustering: " + collection_name + ", Linkage: " + linkage)
    plot_dendrogram(cluster_model, truncate_mode="level", p=5, color_threshold=0.25)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

if __name__ == "__main__":
    main()