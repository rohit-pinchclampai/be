import os
from chunker import load_document, chunk_documents
from embedder import generate_embeddings
from store import upload_embeddings_to_pinecone, query_pinecone, index
from groq_llm import groq_answer

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "/Users/rohitbohra/Desktop/pinchclampai_be/app/services/data/"
NAMESPACE = "docs"
SUPPORTED_EXT = {".pdf", ".txt", ".docx"}
QUERY = "what is the email address of the company?"


def ensure_namespace_exists(namespace: str) -> None:
    """
    Ensure a Pinecone namespace exists (implicit creation if empty).
    Logs if new namespace is being created.
    """
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})

    if namespace not in namespaces or namespaces[namespace].get("vector_count", 0) == 0:
        print(f"ℹ️ Namespace '{namespace}' does not exist yet. It will be created on first upsert.")
    else:
        print(f"✅ Namespace '{namespace}' already exists with {namespaces[namespace]['vector_count']} vectors.")


def process_documents():
    # Ensure namespace
    ensure_namespace_exists(NAMESPACE)

    # Loop through all files
    for file_name in os.listdir(DATA_DIR):
        ext = os.path.splitext(file_name)[-1].lower()
        if ext not in SUPPORTED_EXT:
            continue  # skip unsupported files

        file_path = os.path.join(DATA_DIR, file_name)
        print(f"📄 Processing file: {file_path}")

        # Load the document
        documents = load_document(file_path=file_path)

        # Chunk the document
        chunked_text = chunk_documents(docs=documents)
        print(f"✂️ Number of chunks created: {len(chunked_text)}")

        # Extract raw text
        texts_to_embed = [doc.page_content for doc in chunked_text]

        # Generate embeddings
        embedded_data = generate_embeddings(texts=texts_to_embed)

        # Upload to Pinecone
        upload_embeddings_to_pinecone(embedded_data, namespace=NAMESPACE)
        print(f"✅ Uploaded {len(embedded_data)} vectors from {file_name} into namespace '{NAMESPACE}'\n")


def run_query(query: str):
    retrieved_chunks = query_pinecone(query, top_k=5, namespace=NAMESPACE)
    print(f"🔍 Retrieved {len(retrieved_chunks)} chunks from Pinecone")

    llm_answer = groq_answer(question=query, context="\n".join([m.metadata.get("text", "") for m in retrieved_chunks]))
    print("🤖 LLM Answer:\n", llm_answer)


if __name__ == "__main__":
    process_documents()
    run_query(QUERY)
