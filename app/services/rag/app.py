import os
import tempfile
import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.services.rag.chunker import load_document, chunk_documents
from app.services.rag.embedder import generate_embeddings
from app.services.rag.store import upload_embeddings_to_pinecone, query_pinecone, index
from app.services.rag.groq_llm import groq_answer
from pydantic import BaseModel 


# -----------------------------
# Config
# -----------------------------
NAMESPACE = "demo-docs"
SUPPORTED_EXT = {".pdf", ".txt", ".docx"}
EMBEDDING_TTL_MINUTES = 30  # Time to keep uploaded vectors in Pinecone

# In-memory tracker for uploaded namespaces and expiration
namespace_tracker = {}

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="RAG Backend (Groq + Pinecone)")

# -----------------------------
# CORS Middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pinchclampai.com",
        "https://www.pinchclampai.com",
        "http://localhost:3000",  # local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str] = []

# -----------------------------
# Utilities
# -----------------------------
def ensure_namespace_exists(namespace: str) -> None:
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})
    if namespace not in namespaces or namespaces[namespace].get("vector_count", 0) == 0:
        print(f"ℹ️ Namespace '{namespace}' will be created on first upsert.")
    else:
        print(f"✅ Namespace '{namespace}' already exists with {namespaces[namespace]['vector_count']} vectors.")

async def cleanup_expired_namespaces():
    while True:
        now = datetime.utcnow()
        expired = [ns for ns, exp in namespace_tracker.items() if now >= exp]
        for ns in expired:
            print(f"🗑️ Cleaning up expired namespace: {ns}")
            index.delete(delete_all=True, namespace=ns)
            del namespace_tracker[ns]
        await asyncio.sleep(60)  # Check every minute

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_expired_namespaces())

# -----------------------------
# Upload Endpoint
# -----------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        ensure_namespace_exists(NAMESPACE)

        # Load and chunk document
        documents = load_document(file_path=tmp_file_path)
        chunked_text = chunk_documents(docs=documents)
        texts_to_embed = [doc.page_content for doc in chunked_text]

        # Generate embeddings
        embedded_data = generate_embeddings(texts=texts_to_embed)

        # Upload embeddings to Pinecone
        upload_embeddings_to_pinecone(embedded_data, namespace=NAMESPACE)

        # Set expiration
        namespace_tracker[NAMESPACE] = datetime.utcnow() + timedelta(minutes=EMBEDDING_TTL_MINUTES)

        return {
            "message": f"✅ Uploaded {len(embedded_data)} vectors from {file.filename} into namespace '{NAMESPACE}'",
            "namespace_expiration": str(namespace_tracker[NAMESPACE])
        }

    finally:
        # Cleanup temp file
        os.remove(tmp_file_path)

# -----------------------------
# Query Endpoint
# -----------------------------
@app.post("/query")
async def query_endpoint(query: str = Form(...)):
    """Ask a query against the knowledge base."""
    if NAMESPACE not in namespace_tracker:
        return {"answer": "No documents uploaded yet."}

    retrieved_chunks = query_pinecone(query, top_k=5, namespace=NAMESPACE)

    if not retrieved_chunks:
        return {"answer": "No relevant information found."}

    context = "\n".join([m.metadata.get("text", "") for m in retrieved_chunks])
    llm_answer = groq_answer(question=query, context=context)

    return {
        "query": query,
        "retrieved_chunks": [m.metadata.get("text", "") for m in retrieved_chunks],
        "answer": llm_answer,
    }
# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
