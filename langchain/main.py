from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser

import streamlit as st 
import os 
from dotenv import load_dotenv 

load_dotenv()

# Fix getenv usage
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Prompt Template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

st.title('LangChain Demo with GROQ API')

input_text = st.text_input("Search the topic you want")

llm = ChatGroq(model="llama-3.3-70b-versatile")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
