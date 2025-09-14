import numpy as np 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List , Dict , Any , Tuple 
from sklearn.metrics.pairwise import cosine_similarity


