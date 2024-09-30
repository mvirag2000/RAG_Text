##
## Quantitative analysis of similarity between novels
##
from chroma_utils import DisplayDocs, GetCollection, GetDocById, DisplayOne
import numpy as np
import matplotlib.pyplot as plt

collection_name1 = "tolstoy"
collection_name2 = "thackeray"
collection_path = "chroma2" 
model_name = "text-embedding-3-small"
select = 3
support = 3

def main():

    novel1 = GetCollection(collection_name1, collection_path, model_name)
    novel2 = GetCollection(collection_name2, collection_path, model_name)

    docs1 = novel1.get(include=["embeddings"]) # Docs come back in id-as-string order
    docs2 = novel2.get(include=["embeddings"])

    vectors1 = docs1["embeddings"]
    vectors2 = docs2["embeddings"]

    vectors1 = np.array(vectors1)
    vectors2 = np.array(vectors2)
    dot_products = np.dot(vectors1, vectors2.T)

    ids1 = [str(i) for i in range(vectors1.shape[0])]
    ids1.sort()
    ids1 = np.array(ids1)

    similarity1to2 = np.max(dot_products, axis=1) # Max match with novel2 for each row of novel1 
    similarity2to1 = np.max(dot_products, axis=0) # Max match with novel1 for each column of novel2 
    assert similarity1to2.shape[0] == vectors1.shape[0]
    assert similarity2to1.shape[0] == vectors2.shape[0]

    fig1 = plt.figure(figsize=(11,5))
    ax1 = fig1.add_subplot(121)
    ax1.hist(similarity1to2, bins=200, density=True)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Max similarity in {collection_name2} for chunks of {collection_name1}')
    ax2 = fig1.add_subplot(122)
    ax2.hist(similarity2to1, bins=200, density=True)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Frequency') # Yes I really want to see both charts
    ax2.set_title(f'Max similarity in {collection_name1} for chunks of {collection_name2}')
    plt.show()

    arg1 = similarity1to2.argsort()[::-1]
    sorted_ids1 = ids1[arg1]
    top_five1 = sorted_ids1[:select]
    print(top_five1)

    sorted_ids2 = similarity2to1.argsort()[::-1]
    top_five2 = sorted_ids2[:select]
    print(top_five2)

    sorted_cosines = np.sort(dot_products, axis=0)
    sorted_cosines = np.sort(sorted_cosines, axis=1)[::-1, ::-1]
    top_corner = sorted_cosines[:select, :select]
    np.set_printoptions(precision=2)
    print(top_corner)

    a = sorted_ids1[:1][0]
    select1 = GetDocById(novel1, a, True)
    new_vector1 = select1["embeddings"][0]
    old_vector1 = vectors1[a]
    assert new_vector1 == old_vector1

    select2 = GetDocById(novel2, sorted_ids2[:1][0], True)
    similarity = np.dot(select1["embeddings"][0], select2["embeddings"][0])
    assert similarity == sorted_cosines[:1, :1] 

    DisplayOne(collection_name1, select1)
    DisplayOne(collection_name2, select2)




if __name__ == "__main__":
    main()