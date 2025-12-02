import streamlit as st
from doc_reader import process_all_pds
from chunks import split_documents
from embedding_man import EmbeddingManager
from vec_store import vectorstore
from ll_model import llm   # assuming you define `llm` inside ll_model.py
from rag_ret import *
from rag import *

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="RAG Q&A", page_icon="📚", layout="wide")
st.title("📖 Financial Reports QA Dataset for RAG-based LLM Fin")

# -------------------------------
# Load and Process PDFs (cached)
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_data():
    st.info("📂 Loading and processing PDF documents...")
    
    # Step 1: Read PDFs
    all_pdfs = process_all_pds("../data")
    
    # Step 2: Split into chunks
    chunks = split_documents(all_pdfs)
    
    # Step 3: Generate embeddings
    texts = [doc.page_content for doc in chunks]
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.generate_embeddings(texts)
    
    # Step 4: Store in vector DB
    vs = vectorstore()
    vs.add_documents(chunks, embeddings)
    
    # Step 5: Create retriever + pipeline
    retriever = Rag_retriever(vs, embedding_manager)
    rag_pipeline = AdvancedRAGPipeline(retriever, llm)
    
    return rag_pipeline

rag_pipeline = load_data()

# -------------------------------
# User Interaction
# -------------------------------
st.subheader("💬 Ask a Question")
question = st.text_input("Type your question here:")

if question:
    with st.spinner("🤖 Thinking..."):
        result = rag_pipeline.query(
            question, 
            top_k=3, 
            min_score=0.1, 
            stream=True, 
            summarize=True
        )

    # -------------------------------
    # Results Display
    # -------------------------------
    st.markdown("### ✅ Final Answer")
    st.write(result['answer'])

    st.markdown("### 📝 Summary")
    st.write(result['summary'])

    #st.markdown("### 📜 Conversation History")
    #.write(result['history'][-1])
