import numpy as np 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List , Dict , Any , Tuple 
from sklearn.metrics.pairwise import cosine_similarity
import os 

class vectorstore:
    def __init__(self, collection_name:str = "pdf_documents", persist_directory:str = "../data/vectore_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory , exist_ok=True)
            self.client = chromadb.PersistentClient(path =self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata= {"description":"PDF  document embedding for RAG"}
            )   
            print(f"\n Vector store initialized.Collection:{self.collection_name} \n")
            print(f"\n Existing document in collection: {self.collection.count()} \n")
        
        except Exception as e:
            print(f"\n Error initializing vector store:{e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray, reset: bool = True):
        if len(documents) != len(embeddings):
            raise ValueError('Number of documents must match number of embeddings')

        if reset:
            existing_ids = self.collection.get()["ids"]
            print("\n⚠️ Clearing existing documents from the vector store...\n")
            print(f"\n✅ Vector store has {len(existing_ids)} documents. Clearing...\n")

            if existing_ids:
                self.collection.delete(ids=existing_ids)
                print("🗑️ All documents deleted.\n")

        print(f'Adding {len(documents)} documents to vector store ...')

        ids =[]
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i , (doc,embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f'doc_{uuid.uuid4().hex[:8]}_{i}'
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        
        try:
            self.collection.add(
                ids = ids,
                embeddings = embeddings_list,
                metadatas = metadatas,
                documents = documents_text
            )
            print("✅ Documents added successfully.\n")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

