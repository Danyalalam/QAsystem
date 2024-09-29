# model.py

import os
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex

def get_gemini_model(api_key):
    """
    Initializes the Gemini LLM model.
    
    Args:
        api_key (str): The API key for the Gemini model.
        
    Returns:
        Gemini: An instance of the Gemini LLM model.
    """
    gemini_model = Gemini(model_name="models/gemini-pro", api_key=api_key)
    return gemini_model

def create_index(documents, llm, embed_model):
    """
    Builds a VectorStoreIndex from the provided documents and models.
    
    Args:
        documents (list): The documents to index.
        llm (Gemini): The LLM model.
        embed_model (GeminiEmbedding): The embedding model.
        
    Returns:
        VectorStoreIndex: The created index.
    """
    index = VectorStoreIndex.from_documents(
        documents=documents,
        llm=llm,
        embed_model=embed_model,
        chunk_size=800,
        chunk_overlap=20
    )
    
    # Persist the index for future use
    index.storage_context.persist()
    
    return index

def get_query_engine(index, llm):
    """
    Initializes the query engine with the index and LLM.
    
    Args:
        index (VectorStoreIndex): The index.
        llm (Gemini): The LLM model.
        
    Returns:
        QueryEngine: The query engine instance.
    """
    query_engine = index.as_query_engine(llm=llm)
    return query_engine
