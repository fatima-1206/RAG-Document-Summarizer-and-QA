from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq LLM for summarization
llm_sum = ChatGroq(
    model="llama-3.1-8b-instant",  
    temperature=0.2,
    max_tokens=1024
)

def get_summary(vectorstore="", llm=llm_sum, k:int=20):
    """
    Generate a summary of the most salient document chunks from a vectorstore using Groq API.

    Args:
        All arguments are optional.
        vectorstore (str, optional): The vectorstore to use for retrieving documents. 
        llm (ChatGroq, optional): The language model for summarization. 
        k (int, optional): The number of salient chunks to consider for summarization. Defaults to 20.

    Returns:
        str: The generated summary of the selected document chunks.
    """
    from storage_and_retrieval import db
    vectorstore = db
    collection = vectorstore._collection
    all_docs = collection.get(include=["documents", "embeddings"])
    
    embeddings = np.array(all_docs["embeddings"])
    documents = all_docs["documents"]

    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    salience_scores = np.linalg.norm(embeddings - centroid, axis=1)

    lower_bound = np.percentile(salience_scores, 5)
    upper_bound = np.percentile(salience_scores, 86)

    mid_band_indices = [
        i for i, score in enumerate(salience_scores)
        if lower_bound < score < upper_bound
    ]

    mid_band_indices.sort(key=lambda i: salience_scores[i], reverse=True)
    k = min(k, len(mid_band_indices))
    salient_chunks = [documents[i] for i in mid_band_indices[:k]]
    # if a chunk is larger than 500 characters, get rid of it
    salient_chunks = [chunk for chunk in salient_chunks if len(chunk) <= 500]

    # Combine chunks into context
    text = "\n\n".join(salient_chunks)
    # Limit context to approximately 3000 characters to stay within API limits
    text = text[:3000]
    
    print(f"Number of chunks: {len(salient_chunks)}\n")
    
    # Create a summarization prompt
    summary_prompt = f"""Please provide a concise summary of the following text in 5-6 sentences:

{text}

Summary:"""
    
    summary = llm.invoke(summary_prompt)
    print(summary)
    return summary.content if summary.content else summary


