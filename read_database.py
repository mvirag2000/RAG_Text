##
## Read text and vectors from Chroma DB 
##
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

load_dotenv()
collection_name = "eliot"
DATA_PATH = "data/" + collection_name
# DATA_PATH = "data/eliot" 
CHROMA_PATH = DATA_PATH + "/chroma" 

client = chromadb.Client(Settings(is_persistent=True, persist_directory=CHROMA_PATH))
print(client.list_collections())

collection = client.get_collection(name=collection_name)
document_id = "1"
result = collection.get(
    ids=[document_id],
    include=["embeddings"], # Embeddings are excluded unless you ask for them 
    )
for item in result: print(item, result[item])  
print(len(result["embeddings"][0])) # It looks like the OpenAI default is 1536 
