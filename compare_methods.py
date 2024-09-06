##
## Search Chroma using string versus vector  
##
import numpy as np
from chroma_utils import CreateEmbedding, DisplayDocs, GetCollection 

def main():
    search_string = "Did the Russians set fire to Moscow?"
    tolstoy = GetCollection("tolstoy")
    
    vector = CreateEmbedding(search_string)
    results = tolstoy.query(
        query_embeddings=[vector],
        n_results = 3
        )
    DisplayDocs("Vector Results", results)

    results = tolstoy.query(
        query_texts=[search_string],
        n_results = 3
        )
    DisplayDocs("String Results", results)

if __name__ == "__main__":
    main()