import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def get_openai_api_key():
    """Get OpenAI API key from environment variables"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return api_key
