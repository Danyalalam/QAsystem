# app.py

import streamlit as st
import os
import logging
from dotenv import load_dotenv
from QAWithPDF.data_ingestion import load_documents_from_uploaded_files
from QAWithPDF.embedding import get_gemini_embedding_model
from QAWithPDF.model_api import get_gemini_model, create_index, get_query_engine

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

def main():
    st.title("QASystem - GenAI Application")
    st.write("Upload documents and ask questions based on their content.")

    # Load Google API Key
    google_api_key = os.getenv('GOOGLE_API_KEY')

    # Ensure the API key is present
    if not google_api_key:
        st.error("Google API key is missing. Please check your .env file.")
        return

    # Initialize models
    try:
        st.info("Initializing models...")
        gemini_model = get_gemini_model(api_key=google_api_key)
        gemini_embed_model = get_gemini_embedding_model()
        st.success("Models initialized successfully.")
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        logging.error(f"Error initializing models: {e}")
        return

    # File uploader
    uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=['txt', 'pdf', 'docx'])
    if uploaded_files:
        try:
            st.info("Loading and processing uploaded documents...")
            documents = load_documents_from_uploaded_files(uploaded_files)
            st.success(f"Successfully loaded {len(documents)} documents.")

            # Create the index
            try:
                st.info("Creating index...")
                index = create_index(documents=documents, llm=gemini_model, embed_model=gemini_embed_model)
                query_engine = get_query_engine(index=index, llm=gemini_model)
                st.success("Index created successfully.")
            except Exception as e:
                st.error(f"Error creating index: {e}")
                logging.error(f"Error creating index: {e}")
                return

            # Question input and submit button
            user_query = st.text_input("Ask a question about your documents:")
            if st.button("Submit"):
                if user_query.strip() == '':
                    st.warning("Please enter a question.")
                else:
                    try:
                        response = query_engine.query(user_query)
                        st.markdown(f"**Response:** {response}")
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        logging.error(f"Error processing query: {e}")
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            logging.error(f"Error loading documents: {e}")

if __name__ == "__main__":
    main()
