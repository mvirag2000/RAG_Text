##
## Read text and vectors from Chroma DB 
##
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chroma_utils import DisplayOne, GetCollection, GetDocById
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

collection_name = "eliot"
collection_path = "chroma3" 

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
    # model_name doesn't matter because no new embedding here 
    print("Collection: " + collection_name)

    collection = GetCollection(collection_name, collection_path)
    count = collection.count()
    print(f"Documents: {count:,}")

    doc = GetDocById(collection, 0, True)
    dims = len(doc["embeddings"][0])
    print(f"Dimensions: {dims:,}")

    docs = collection.get(include=["embeddings"])
    vectors = docs["embeddings"]
    ids = docs["ids"]

    
    DisplayOne("Some doc", GetDocById(collection, 450))
    

    pca = PCA(n_components=10)
    reduced = pca.fit_transform(vectors)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())

  

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

    clusters = AgglomerativeClustering(n_clusters=5, metric='cosine', linkage='complete', compute_distances=True)
    clusters.fit(vectors)

    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(clusters, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()




if __name__ == "__main__":
    main()