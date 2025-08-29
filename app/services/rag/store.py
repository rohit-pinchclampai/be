from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
from app.services.rag import config
from app.services.rag.embedder import create_embeddings

pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Ensure index exists
if config.INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=config.INDEX_NAME,
        dimension=config.DIMENSION,
        metric=config.METRIC,
        spec=ServerlessSpec(
            cloud=config.CLOUD,
            region=config.REGION
        )
    )
index = pc.Index(config.INDEX_NAME)

def upload_embeddings_to_pinecone(vectors: List[Dict], namespace: str = "docs"):
    """Upload embedding vectors into Pinecone."""
    index.upsert(vectors=vectors, namespace=namespace)
    print(f"✅ Uploaded {len(vectors)} vectors.")

def query_pinecone(query: str, top_k: int = 5, namespace: str = "docs"):
    """Query Pinecone for relevant vectors."""
    query_vector_emb = create_embeddings([query])
    res = index.query(
        vector=query_vector_emb,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    return res.matches
