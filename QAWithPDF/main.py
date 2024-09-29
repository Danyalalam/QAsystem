# main.py

import os
from dotenv import load_dotenv
from data_ingestion import load_documents
from embedding import get_gemini_embedding_model
from model_api import get_gemini_model, create_index, get_query_engine

# Load environment variables
load_dotenv()

# Load Google API Key
google_api_key = os.getenv('GOOGLE_API_KEY')

# Load documents
documents = load_documents(input_dir='../data')

# Initialize models
gemini_model = get_gemini_model(api_key=google_api_key)
gemini_embed_model = get_gemini_embedding_model()

# Create the index
index = create_index(documents=documents, llm=gemini_model, embed_model=gemini_embed_model)

# Create query engine
query_engine = get_query_engine(index=index, llm=gemini_model)

# Example query
response = query_engine.query("What projects are mentioned?")
print(response)
