from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq LLM for query responses
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # or "mixtral-8x7b-32768"
    temperature=0.7,
    max_tokens=512
)


def trim_context_to_token_limit(docs, max_chars:int = 3000):
    """
    Trims the provided documents so that their combined character count does not exceed max_chars.

    Args:
        docs (list): List of documents, each with or without a 'page_content' attribute or as a string.
        max_chars (int): Maximum allowed number of characters in the combined context. Defaults to 3000.

    Returns:
        str: Concatenated document texts, trimmed to fit within the character limit.
    """
    context = ""
    total_chars = 0

    for doc in docs:
        doc_text = doc.page_content.strip() if hasattr(doc, 'page_content') else str(doc)
        if total_chars + len(doc_text) <= max_chars:
            context += doc_text + "\n\n"
            total_chars += len(doc_text)
        else:
            break
    return context.strip()

def get_query_response(query:str)-> str:
    """
    Processes a user query by retrieving relevant documents, constructing a prompt,
    and generating a response using the Groq API.

    Args:
        query (str): The user's question.

    Returns:
        str: The generated answer based on retrieved context.
    """
    from storage_and_retrieval import db
    from storage_and_retrieval import retriever
    results = retriever.get_relevant_documents(query)
    context = trim_context_to_token_limit(results, max_chars=3000)
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following context to answer the question at the end. 
        Even if the question is not directly answered in the context, give an answer based on your knowledge.
        Give a detailed answer based on the context provided. Don't say anything else only the answer itself.
        Context:
        {context}

        Question:
        {question}

        Answer:"""
    )
    print("Context length:", len(context.split()))

    # Format the prompt with context and query
    formatted_prompt = prompt_template.format(context=context, question=query)
    
    print("Running RAG query with Groq API...")
    response = llm.invoke(formatted_prompt)

    return response.content if response.content else response


