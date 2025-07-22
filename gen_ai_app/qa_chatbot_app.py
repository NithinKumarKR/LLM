import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable  # optional

# Load environment variables
load_dotenv()
gpk = os.getenv("GPK")
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
project_name = os.getenv("LANGCHAIN_PROJECT", "Groq_ChatBot_Project")

# LangSmith setup
if langsmith_api_key:
    from langchain.callbacks.tracers.langchain import LangChainTracer
    tracer = LangChainTracer(project_name=project_name)
else:
    tracer = None

# Streamlit UI
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")
st.title("🤖 Q&A Chatbot using Groq + LangChain + LangSmith")

# Sidebar configs
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1024, 300)
model_name = st.sidebar.selectbox("Choose Model", ["gemma2-9b-it", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0)

# Memory setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Prompt template with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
])

llm = ChatGroq(
    model=model_name,
    groq_api_key=gpk,
    temperature=temperature,
    max_tokens=max_tokens
)

output_parser = StrOutputParser()

chain: Runnable = prompt | llm | output_parser

@traceable(name="generate_response", project_name=project_name)
def generate_response(question: str):
    inputs = {
        "question": question,
        "chat_history": st.session_state.chat_history
    }
    if tracer:
        return chain.invoke(inputs, config={"callbacks": [tracer]})
    return chain.invoke(inputs)

# Display existing chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat input
user_query = st.chat_input("Ask your question")
if user_query:
    st.chat_message("user").write(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.spinner("Generating response..."):
        try:
            response = generate_response(user_query)
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append(AIMessage(content=response))
        except Exception as e:
            st.error(f"Error: {e}")
