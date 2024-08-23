import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma 
import chromadb

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """Answer this question {question} including the following context: {context}"""

def main():
    embedding_function = OpenAIEmbeddings()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = ChatOpenAI()
    query_text = input("Enter query: ")
    while (query_text != "quit"):   

        # Search RAG database for context
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)       
        results = db.similarity_search_with_relevance_scores(query_text, k=10)

        if len(results) == 0 or results[0][1] < 0.7:
            print(f"Unable to find matching results.")
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt = prompt_template.format(context=context_text, question=query_text)      
            response = model.invoke(prompt)
            sources = [doc.metadata.get("source", None) for doc, _score in results]
            print("\nCONTEXT: " + context_text + "\n")
            print("\nRESPONSE: " + response.content + "\n")
            # print(sources)

        query_text = input("Enter query: ")

if __name__ == "__main__":
    main()
