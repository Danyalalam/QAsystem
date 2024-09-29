from llama_index.core import SimpleDirectoryReader
import sys
from exception import customexception
from logger import logging

def load_data(data):
    """
    Load PDF documents from a directory
    
    parameters:
    data: str: path to the directory containing PDF documents
    
    returns:
    list: list of PDF documents
    
    
    """
    try:
        logging.info("Loading PDF documents from directory")
        loader = SimpleDirectoryReader("data")
        documents = loader.load_data()
        logging.info("PDF documents loaded successfully")
        return documents
    except Exception as e:
        logging.info("Error loading PDF documents")
        raise customexception(e,sys)