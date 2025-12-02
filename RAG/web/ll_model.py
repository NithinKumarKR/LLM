# Integration Vectordb Context pipeline with LLm Output
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

## Initialize the Groq LLM

groq_api_key =  os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key =groq_api_key , model_name = "gemma2-9b-it", temperature=.1 , max_tokens=1024)