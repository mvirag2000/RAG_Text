##
## Read text and vectors from Chroma DB 
##
from dotenv import load_dotenv
from chroma_utils import CreateEmbedding, DisplayDocs, GetCollection, GetDocById, BruteScan
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
As context for the question that follows, here are some passages from the novel Vanity Fair:

{context1}

As further context, here are some passages from the novel War and Peace:

{context2}

Compare the two novels, with reference to the context provided.
"""


def main():
    n = 3
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = ChatOpenAI(model='gpt-3.5-turbo')

    tolstoy = GetCollection("tolstoy", "chroma2")
    # eliot = GetCollection("eliot")
    thackeray = GetCollection("thackeray", "chroma2")

    for i in range(thackeray.count()):
        results1 = GetDocById(thackeray, i, True)
        query_vec = results1["embeddings"][0]    
        results2 = tolstoy.query(
            query_embeddings=query_vec,
            n_results = n)
        if (min(results2['distances'][0]) < 0.31): # Distance NOT similarity 
            DisplayDocs("Tolstoy Results", results2)
            more_results = GetDocById(tolstoy, results2["ids"][0][0], True)

            # Go get three from the other side and overwrite results1
            second_vec = more_results["embeddings"][0]  
            results1 = thackeray.query(
                query_embeddings=second_vec,
                n_results = n)
            DisplayDocs("Thackeray Results", results1)
            context_text1 = "\n\n".join(results1['documents'][0])
            context_text2 = "\n\n".join(results2['documents'][0])
            prompt = prompt_template.format(context1=context_text1, context2=context_text2)  
            print(prompt) # Limit for gpt-3.5-turbo is 4000 
            response = model.invoke(prompt)
            print("\n" + response.content + "\n")


if __name__ == "__main__":
    main()