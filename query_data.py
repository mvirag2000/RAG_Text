import argparse
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma 
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

CHROMA_PATH = "data/tolstoy/chroma"

PROMPT_TEMPLATE = """Answer this question: {question} based on these excerpts: {context}"""

model_name = "text-embedding-3-large" 

def show_text(title, text):
    window = tk.Tk()
    window.title(title)
    window.geometry("650x300") 
    text_widget = tk.Text(window, wrap="word", font=("Arial", 10))
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(window, orient="vertical", command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    text_widget.config(yscrollcommand=scrollbar.set)
    text_widget.insert(tk.END, text)
    window.mainloop()

def main():
    embedding_function = OpenAIEmbeddings(
        model=model_name,
        deployment=model_name
    )
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # model = ChatOpenAI()
 
    query_text = input("Enter query: ")
    while (query_text != "quit"):   

        # Search RAG database for context
        db = Chroma(persist_directory=CHROMA_PATH, 
            embedding_function=embedding_function,
        )
        results = db.similarity_search_with_relevance_scores(query_text, k=10)

        if len(results) == 0 or results[0][1] < 0.7:
            print(f"Unable to find matching results.")
            stop
        else:
            context_text = "\n---\n".join([doc.page_content for doc, _score in results])
            prompt = prompt_template.format(context=context_text, question=query_text)      
            # sources = [doc.metadata.get("source", None) for doc, _score in results]
            source_listing = [doc.metadata.get("source", None) + "  {:4.2f}".format(_score) + "\n" + doc.page_content + "\n\n" for doc, _score in results]
            # response = model.invoke(prompt)
            # print("\n" + response.content + "\n")
            show_text("Source Data", source_listing)

        query_text = input("Enter query: ")

if __name__ == "__main__":
    main()
