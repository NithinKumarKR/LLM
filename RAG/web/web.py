import streamlit as st
from naukri_scraper import *
from chunks import split_documents
from embedding_man import EmbeddingManager
from vec_store import vectorstore
from ll_model import llm 
from rag_ret import *
from rag import *
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
import datetime


# Example usage
page_url = "https://www.investorgain.com/report/live-ipo-gmp/331/ipo/"
docs = get_docs(page_url)

chunks = split_documents(docs)

texts = [doc.page_content for doc in chunks]
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings(texts)

# Step 4: Store in vector DB
vs = vectorstore()
vs.add_documents(chunks, embeddings)

# Step 5: Create retriever + pipeline
retriever = Rag_retriever(vs, embedding_manager)
rag_pipeline = AdvancedRAGPipeline(retriever, llm)

question = f"""
You are an expert financial assistant. I have a list of IPOs with their details including name, price, subscription %, subscription ratio, open date, close date, and market capitalization.

Today's date is {datetime.datetime.today().strftime('%d-%b')}.

Please answer the following question:

"I need the IPOs which are active today with all their details."

Rules:
1. Only include IPOs that are currently active today — i.e., the current date is **between the open date and close date**. Ignore any other visual markers like text color, background, or emojis.
2. Include all available details for each IPO: name, price, subscription %, subscription ratio, open date, close date, and market capitalization.  
3. Present the data in a clear, readable format, like a table or structured list.  
4. Do not include IPOs that are not active today or already closed.  

Provide the answer directly without extra explanation.
"""

result = rag_pipeline.query( question,  top_k=3,  min_score=0.1,  stream=True, summarize=True )


print(result['answer'])
