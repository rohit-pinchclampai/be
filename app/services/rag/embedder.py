import voyageai
import os
import uuid
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
voyage_api_key = os.getenv("VOYAGE_API_KEY")


def create_embeddings(texts: List[str], model: str = "voyage-3-large"):
    """
    
    """
    client = voyageai.Client(api_key=voyage_api_key)

    # 2. Get embeddings
    emb_obj = client.embed(
        texts=texts,
        model=model
    )
    return emb_obj.embeddings

def generate_embeddings(texts: List[str]) -> List[Dict]:
    """
    Generate embeddings for a list of texts and prepare them for Pinecone upsert.

    Args:
        texts: List of strings to embed.
        model: Embedding model name.

    Returns:
        List of dicts ready for Pinecone upsert.
    """
    print("create_embeddings ->\n\n", create_embeddings(texts))
    embeddings_list = create_embeddings(texts=texts)
    # 3. Format for Pinecone
    vectors = []
    for text, vector in zip(texts, embeddings_list):
        vectors.append({
            "id": str(uuid.uuid4()),  # required by Pinecone
            "values": vector,
            "metadata": {"text": text}
        })

    return vectors


# Example usage
if __name__ == "__main__":
    texts = ["Hello world", "This is a RAG pipeline example."]

    vectors = generate_embeddings(texts)

    # Now you can directly push into Pinecone:
    # index.upsert(vectors=vectors)
    print(vectors[0])  # show one example
    print("✅ Generated", len(vectors), "vectors")
