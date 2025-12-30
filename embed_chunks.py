import json
import os
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

# Directories
CHUNKS_DIR = Path("text_chunks_json")
INDEX_DIR = Path("vector_index")
INDEX_DIR.mkdir(exist_ok=True)

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072


def load_chunks():
    """Load all chunked JSONs into a list of {id, text, metadata} dicts."""
    data = []
    for file in CHUNKS_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            for chunk in chunks:
                data.append({
                    "id": f"{file.stem}_{chunk['chunk_id']}",
                    "text": chunk["text"],
                    "metadata": {
                        "seminar": chunk.get("seminar"),
                        "lecon": chunk.get("lecon"),
                        "pages": chunk.get("pages")
                    }
                })
    return data


def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def build_faiss_index(chunks):
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = []
    ids = []
    embeddings = []

    for chunk in tqdm(chunks, desc="Embedding chunks"):
        embedding = get_embedding(chunk["text"])
        embeddings.append(embedding)
        metadata.append(chunk["metadata"])
        ids.append(chunk["id"])

    embeddings_np = np.array(embeddings).astype("float32")
    index.add(embeddings_np)

    faiss.write_index(index, str(INDEX_DIR / "lacan.index"))

    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "metadata": metadata},
                  f, ensure_ascii=False, indent=2)

    print(f"FAISS index saved with {len(chunks)} entries.")


if __name__ == "__main__":
    print("Loading text chunks...")
    chunks = load_chunks()
    print(f"{len(chunks)} chunks loaded.")
    build_faiss_index(chunks)
