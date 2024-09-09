from chromadb.utils import embedding_functions 
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb
from chromadb.config import Settings
from chroma_utils import DisplayDocs, num_tokens_from_string

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
model_name = "text-embedding-ada-002" 
query_text = 'Is there intrigue over the beneficiary of a wealthy person''s will?'
# query_text = "Are there examples of women being unfaithful?"

CHROMA_PATH1 = "data/eliot/chroma3"
collection_name1 = 'eliot'

CHROMA_PATH2 = "data/tolstoy/chroma3"
collection_name2 = 'tolstoy'

PROMPT_TEMPLATE = """
As context for the question that follows, here are some passages from the novel Middlemarch:

{context1}

As further context, here are some passages from the novel War and Peace:

{context2}

Compare the two novels, with reference to the context provided.
"""

def main():

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = ChatOpenAI(model='gpt-3.5-turbo')
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=model_name)

    client1 = chromadb.Client(Settings(is_persistent=True, persist_directory=CHROMA_PATH1))
    collection1 = client1.get_collection(name=collection_name1, embedding_function=openai_ef)
    results1 = collection1.query(
        query_texts=[query_text],
        n_results = 3
    )
    client2 = chromadb.Client(Settings(is_persistent=True, persist_directory=CHROMA_PATH2))
    collection2 = client2.get_collection(name=collection_name2, embedding_function=openai_ef)
    results2 = collection2.query(
        query_texts=[query_text],
        n_results = 3
    )

    if len(results1) == 0 or results1['distances'][0][0] > 0.40 or \
        len(results2) == 0 or results2['distances'][0][0] > 0.40:
        print(f"Unable to find matching results.")
    else:
        context_text1 = "\n\n".join(results1['documents'][0])
        context_text2 = "\n\n".join(results2['documents'][0])
        prompt = prompt_template.format(context1=context_text1, context2=context_text2) #, question=query_text)     
        print(prompt) # Limit for gpt-3.5-turbo is 4000 
        print(num_tokens_from_string(prompt))
        response = model.invoke(prompt)
        print("\n" + response.content + "\n")
        # DisplayDocs("Source Data", results1)


if __name__ == "__main__":
    main()
