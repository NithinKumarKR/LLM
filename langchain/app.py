from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
import uvicorn
import streamlit as st 
import os 
from dotenv import load_dotenv 

load_dotenv()

# Fix getenv usage
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title= "Langchain Server",
    version= "1.0" ,
    description= "A Simple API Server"
)

add_routes(
    app,
    ChatGroq(    model="llama-3.3-70b-versatile"),
    path ='/groq_api'
)

llm =ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature = 0,
    max_tokens = None ,
    timeout = None ,
    groq_api_key = os.environ['LANGCHAIN_API_KEY'],
    max_retries= 2,
)

prompt = ChatPromptTemplate.from_template('Write me a essay  about {topic}  with 100 words')

add_routes(
    app,
    prompt|llm ,
    path = '/eassy'
)


if __name__ == "__main__":
    uvicorn.run(app, host = '127.0.0.2',port=8001)