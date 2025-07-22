import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub

# Load environment variables
load_dotenv()

# Hugging Face Token
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not huggingface_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not set in the .env file.")

# LangSmith Tracking (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple QA With HuggingFace"

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Question: {question}")
])

# Function to generate response
def generate_rep(question, model_name, temperature, max_tokens):
    llm = HuggingFaceHub(
        repo_id=model_name,
        huggingfacehub_api_token=huggingface_token,
        model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens}
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"question": question})

# Streamlit UI
st.title("Q&A Chatbot using Hugging Face")

# Sidebar options
model_name = st.sidebar.selectbox(
    "Choose a Hugging Face Model",
    [
        "meta-llama/Llama-3-8b-chat-hf",
        "HuggingFaceH4/zephyr-7b-beta",
        "bigscience/bloomz-560m"
    ]
)

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Input and response
user_input = st.text_input("You:")
if user_input:
    with st.spinner("Generating response..."):
        try:
            response = generate_rep(user_input, model_name, temperature, max_tokens)
            st.markdown("**Assistant:** " + response)
        except Exception as e:
            st.error(f"Error generating response: {e}")
else:
    st.info("Awaiting your question...")
