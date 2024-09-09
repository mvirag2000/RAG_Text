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

# CHROMA_PATH = "data/tolstoy/chroma3"
# collection_name = 'tolstoy'
# query_text = 'Did the Russians set fire to Moscow?'
# query_text = 'How many siblings does Natasha have?'
# query_text = 'Why doesn''t Andrew marry Natasha?'
# query_text = 'How is Sonya related to the Rostovs?'

CHROMA_PATH = "data/eliot/chroma3"
collection_name = 'eliot'
query_text = 'Is there intrigue over the beneficiary of a wealthy person''s will?'

model_name = "text-embedding-ada-002" 
PROMPT_TEMPLATE = """
Here is some context for the question that follows:

{context}


Please answer this question based on the above context: {question}
"""

def main():

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = ChatOpenAI(model='gpt-3.5-turbo')

    client = chromadb.Client(Settings(is_persistent=True, persist_directory=CHROMA_PATH))
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=model_name)
    collection = client.get_collection(name=collection_name, embedding_function=openai_ef)

    results = collection.query(
        query_texts=[query_text],
        n_results = 10
        )

    if len(results) == 0 or results['distances'][0][0] > 0.40:
        print(f"Unable to find matching results.")
    else:
        context_text = "\n\n".join(results['documents'][0])
        prompt = prompt_template.format(context=context_text, question=query_text)     
        print(prompt) # Limit for gpt-3.5-turbo is 4000 
        print(num_tokens_from_string(prompt))
        response = model.invoke(prompt)
        print("\n" + response.content + "\n")
        DisplayDocs("Source Data", results)


if __name__ == "__main__":
    main()
