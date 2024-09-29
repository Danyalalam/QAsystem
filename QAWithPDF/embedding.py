# embedding.py

from llama_index.embeddings.gemini import GeminiEmbedding

def get_gemini_embedding_model():
    """
    Initializes the Gemini embedding model.
    
    Returns:
        GeminiEmbedding: An instance of the GeminiEmbedding model.
    """
    gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
    return gemini_embed_model
