
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import sentence_transformers
import time
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Vector creation function with debugging
def create_vs():
    if "vectors" not in st.session_state:
        st.write("Loading and processing documents...")

        try:
            # Load embeddings model
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.embeddings = embedding_model

            # Load PDF documents
            #base_dir = os.path.dirname(os.path.abspath(__file__))
            #pdf_dir = os.path.join(base_dir, "papers")
            #st.session_state.loader = PyPDFDirectoryLoader(pdf_dir)
            st.session_state.loader = DirectoryLoader(
                "C:/Users/nithi/Desktop/GitHub/LLM/GEN AI APP/papers",
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            st.session_state.docs = st.session_state.loader.load()
            st.write(f"✅ Loaded {len(st.session_state.docs)} documents")
            for i, doc in enumerate(st.session_state.docs):
                st.write(f"📄 Document {i+1}: {doc.metadata.get('source', 'Unknown')} - {len(doc.page_content)} characters")

            if not st.session_state.docs:
                st.error("❌ No documents loaded. Check the /papers directory.")
                return

            # Split documents
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.write(f"✅ Split into {len(st.session_state.final_documents)} text chunks")

            # Test embedding generation
            test_text = st.session_state.final_documents[0].page_content
            emb_test = st.session_state.embeddings.embed_query(test_text)
            st.write(f"✅ Sample embedding length: {len(emb_test)}")

            # Create vector store
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
            st.success("✅ Vector database is ready")

        except Exception as e:
            st.error(f"❌ Error during vector creation: {e}")

# Streamlit UI
st.title("GROQ PDF Q&A App")

user_prompts = st.text_input("Enter your query from the papers")

if st.button("Document Embedding"):
    create_vs()

if user_prompts:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create the vector store first by clicking the 'Document Embedding' button.")
    else:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            ret_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = ret_chain.invoke({'input': user_prompts})
            duration = time.process_time() - start

            st.write(f"⏱️ Response Time: {duration:.2f} seconds")
            st.write("💬 Answer:", response['answer'])

            with st.expander("🧠 Document Similarity Context"):
                for i, doc in enumerate(response.get('context', [])):
                    st.write(f"🔹 Chunk {i+1}")
                    st.write(doc.page_content)
                    st.write("---")

        except Exception as e:
            st.error(f"❌ Error during query processing: {e}")
