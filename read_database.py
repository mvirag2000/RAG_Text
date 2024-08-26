##
## Read text and vectors from Chroma DB 
##
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

load_dotenv()
CHROMA_PATH = "chroma" 
collection_name = "tolstoy"

client = chromadb.Client(Settings(is_persistent=True, persist_directory=CHROMA_PATH))
collection = client.get_collection(name=collection_name)
document_id = "1"
result = collection.get(
    ids=[document_id],
    include=["embeddings"], # Embeddings are excluded unless you ask for them 
    )
for item in result: print(item, result[item])  
