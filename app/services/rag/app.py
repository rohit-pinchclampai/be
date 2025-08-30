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

# ------------ Config ------------
APP_NAME = "PinchClampAI RAG API - Render Ready"
VERSION = "0.1.0"

# if you really need to restrict file types, edit this set
ALLOWED_EXT = {".pdf", ".txt", ".docx", ".doc"}

# in-memory flag so /query can warn if nothing uploaded yet (replace with your own logic)
_last_upload_at: Optional[datetime] = None

app = FastAPI(title=APP_NAME, version=VERSION)

# ------------ CORS ------------
# IMPORTANT: list explicit origins (Render/Cloudflare can be picky).
ALLOWED_ORIGINS = [
    "https://pinchclampai.com",
    "https://www.pinchclampai.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,     # only valid with explicit origins (not "*")
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Optional: very defensive preflight fallback (CORS middleware usually covers this)
@app.options("/{rest_of_path:path}")
async def options_cors_preflight(rest_of_path: str, request: Request):
    return JSONResponse(status_code=204, content=None)


# ------------ Models ------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# ------------ Routes ------------
@app.get("/health")
async def health():
    return {"status": "ok", "app": APP_NAME, "version": VERSION}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a file. This demo stores it temporarily and pretends to index it.
    Replace the 'process_file' section with your real RAG pipeline.
    """
    import os as _os  # local alias just to emphasize we run here safely

    # Validate extension
    name = file.filename or ""
    ext = _os.path.splitext(name)[1].lower()
    if ext and ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save to a temp file (so large uploads don’t live in memory)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".bin") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save upload: {e}")

    # --- Your indexing logic here ---
    # e.g. load -> chunk -> embed -> upsert to vector db
    # For now, just mark that we uploaded something.
    global _last_upload_at
    _last_upload_at = datetime.utcnow()

    # Clean up temp file if you don’t need it
    try:
        _os.remove(tmp_path)
    except Exception:
        pass  # not fatal

    return {
        "message": f"✅ File '{name}' received and processed.",
        "uploaded_at": _last_upload_at.isoformat() + "Z",
    }


@app.post("/query")
async def query(req: QueryRequest):
    """
    Accepts JSON: { "query": "...", "top_k": 5 }
    Returns: { "answer": "..." }
    """
    if not _last_upload_at:
        # Safe, friendly message if no file has been uploaded since server start
        return {"answer": "⚠️ No documents uploaded yet. Please upload a file first."}

    # ---- Your real retrieval + LLM answer goes here ----
    # For now, return a mock answer to prove wiring is correct.
    answer = f"Mock answer for: '{req.query}'. (top_k={req.top_k})"
    return {"answer": answer}


# Local dev runner (Render uses Procfile below)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
