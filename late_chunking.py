##
## Working from the Weaviate and Jina examples (both) to load Chroma DB
##
import spacy
from spacy.tokens import Doc
from transformers import AutoModel
from transformers import AutoTokenizer
import numpy as np
import shutil
import os
import chromadb
from chromadb.config import Settings

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

collection_name = "tolstoy"
collection_path = "chroma6" 
model_name = "jinaai/jina-embeddings-v2-base-en"

def sentence_chunker(document, batch_size=None):
    """
    Given a document (string), return the sentences as chunks and span annotations (start and end indices of chunks).
    Using spacy to do this sentence chunking.
    """

    if batch_size is None:
        batch_size = 10000 # no of characters

    # Batch with spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer", config={"punct_chars": None})
    doc = nlp(document)

    docs = []
    for i in range(0, len(document), batch_size):
        batch = document[i : i + batch_size]
        docs.append(nlp(batch))

    doc = Doc.from_docs(docs)

    span_annotations = []
    chunks = []
    for i, sent in enumerate(doc.sents):
        span_annotations.append((sent.start, sent.end))
        chunks.append(sent.text)

    return chunks, span_annotations

def late_chunking(model_output: 'BatchEncoding', span_annotation: list, max_length=None):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs

def main():
    source = "data/" + collection_name + "/Chapter_4.txt"
    path = "data/" + collection_name + "/" + collection_path 
    book = open(source, 'r')
    document = book.read()
    chunks, span_annotations = sentence_chunker(document)
    inputs = tokenizer(document, return_tensors='pt')
    model_output = model(**inputs)
    embeddings = late_chunking(model_output, [span_annotations])[0]

    # Clear directory or collection
    if os.path.exists(path):
        shutil.rmtree(path)

    client = chromadb.Client(Settings(is_persistent=True, persist_directory=path))

    collection = client.create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"}, 
        # embedding_function is optional 
    )
    collection.add(
        documents = chunks, # This line throws error 400 if docs is too big (length or chunk size)
        metadatas = [{"start": s[0], "end": s[1]} for s in span_annotations],
        embeddings = [e.tolist() for e in embeddings],
        ids=[str(i) for i in range(len(chunks))],
    )
    print(f"Saved {len(chunks)} chunks to {path}.")
    readme = open(path + "\\readme.txt", 'w')
    readme.write("Created using native Chroma with " + model_name)
    readme.close()

if __name__ == "__main__":
    main()