import numpy as np 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List , Dict , Any , Tuple 
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingManager:

    def __init__(self, model_name: str ="all-MiniLM-L6-v2"):
        self.model_name =model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"Loading embedding model:{self.model_name}")
            self.model =SentenceTransformer(self.model_name)
            print(f'Model loaded successfully. Embedding dimention:{self.model.get_sentence_embedding_dimension()}')
        except Exception as e:
            print(f'Error loading model {self.model_name}:{e}')
            raise
                  
    def generate_embeddings(self, texts : List[str])->np.array:
        if not self.model:
            raise ValueError('Model not loaded')
        print(f'Generating embeddings for {len(texts)} texts')
        Embedding =self.model.encode(texts, show_progress_bar=True)
        print(f'Generating embeddings with shape:{Embedding.shape}')
        return Embedding
    
    def get_embedding_dimension(self) -> int:
        if not self.model:
            raise ValueError("Module not Loaded")
        return self.model.get_sentence_embedding_dimension()
    
#Embedding_Manager =EmbeddingManager()
#Embedding_Manager