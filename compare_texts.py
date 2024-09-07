##
## Read text and vectors from Chroma DB 
##
from dotenv import load_dotenv
from chroma_utils import CreateEmbedding, DisplayDocs, GetCollection, GetDocById, BruteScan

def main():
    tolstoy = GetCollection("tolstoy")
    eliot = GetCollection("eliot")
    for i in range(eliot.count()):
        doc = GetDocById(eliot, i, True)
        query_vec = doc["embeddings"][0]    
        results = tolstoy.query(
            query_embeddings=query_vec,
            n_results = 3)
        if (min(results['distances'][0]) < 0.40): # Distance NOT similarity 
            print(doc["metadatas"][0])
            print(doc["documents"][0])
            print()
            DisplayDocs("Tolstoy Results", results)

if __name__ == "__main__":
    main()