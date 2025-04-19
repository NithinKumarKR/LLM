from dotenv import load_dotenv
from langchain_ollama import OllamaLLM  # Corrected import for the new class
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
load_dotenv()

LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="pr-crushing-west-73"

##  Prompt Template

promt = ChatPromptTemplate.from_messages(
    [
        ("system", "Your are a helpful assistant. Please respond"),
        ("user", "Question:{question}")
    ]
)

## Streamlit Framework

st.title("Langchain Demo with LLM")
input_text = st.text_input("How can I help you?")



## Ollama model initialization

llm = OllamaLLM(model="gemma:2b")  # Using the new OllamaLLM class

output_parser = StrOutputParser()

chain = promt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))