

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from risk_knowledge import risk_texts

model = SentenceTransformer("intfloat/multilingual-e5-large")
embeddings = model.encode([f"passage: {text}" for text in risk_texts])
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def search_risk_explanation(query, top_k=1):
    query_vec = model.encode(f"query: {query}").reshape(1, -1)
    D, I = index.search(query_vec, top_k)
    return [risk_texts[i] for i in I[0]]