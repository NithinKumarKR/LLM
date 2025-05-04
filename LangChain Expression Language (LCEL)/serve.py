import langserve.pydantic_v1 as pydantic_v1  # Ensure LangServe uses Pydantic v1
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langserve import add_routes
from langchain_groq.chat_models import ChatGroq
import uvicorn
import os
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
import aiohttp

# Ensure Pydantic v1 is being used by LangServe
import langserve.pydantic_v1 as pydantic_v1
pydantic_v1  # This ensures LangServe uses Pydantic v1

# Load .env configuration file
load_dotenv()


# Setup the Groq model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Define the generic template
generic_template = "Translate the following into {language}:"

# Create the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [("system", generic_template), ("user", "{text}")]
)

# Set up the output parser
parser = StrOutputParser()

# Pydantic model for the TranslationRequest
class TranslationRequest(BaseModel):
    language: str
    text: str

    class Config:
        arbitrary_types_allowed = True  

