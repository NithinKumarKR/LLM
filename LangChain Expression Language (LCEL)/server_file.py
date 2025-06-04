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
from typing import Dict

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

# Pydantic model for the TranslationRequest for the /translate endpoint
class TranslationRequest(BaseModel):
    language: str
    text: str

# Define the input type for the chain explicitly
class ChainInput(BaseModel):
    language: str
    text: str

class ChainOutput(BaseModel):
    output: str

# Define the LangChain chain with explicit input type

chain = (
    RunnablePassthrough.assign(language=lambda x: x["language"], text=lambda x: x["text"])
    | prompt
    | model
    | parser
    | (lambda x: {"output": x})  # Wrap the string output in a dictionary
).with_types(input_type=ChainInput, output_type=ChainOutput)


# Function to fetch translation from Groq API (remains the same)
async def fetch_translation_from_groq(text: str, language: str):
    url = "https://groq.ai/api/v1/translate"  # Replace with actual Groq API endpoint
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"text": text, "language": language, "api_key": groq_api_key}) as response:
            return await response.json()

# FastAPI app definition
app = FastAPI(title="LC Serve", version="1.0", description="A simple API Server using Langchain runnable interfaces")

# Add routes to the FastAPI app
add_routes(app, chain, path="/chain")

@app.post("/translate")
async def translate(request: TranslationRequest):
    # Call the Groq API to perform the translation
    result = await fetch_translation_from_groq(request.text, request.language)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)