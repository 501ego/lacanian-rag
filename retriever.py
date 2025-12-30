import json
import os
from pathlib import Path
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

# Paths
INDEX_PATH = Path("vector_index/lacan.index")
METADATA_PATH = Path("vector_index/metadata.json")

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

# Load FAISS index and metadata
index = faiss.read_index(str(INDEX_PATH))
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
ids = meta["ids"]
metadata = meta["metadata"]


def embed_query(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)


def search_similar_chunks(query: str, top_k: int = 5):
    query_vector = embed_query(query)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(metadata):
            continue
        result = {
            "score": float(distances[0][i]),
            "id": ids[idx],
            "seminar": metadata[idx]["seminar"],
            "lecon": metadata[idx]["lecon"],
            "pages": metadata[idx]["pages"],
        }
        results.append(result)

    return results
