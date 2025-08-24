from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from schemas import QueryRequest, QueryResponse
from services.chunker import chunk_text
from .utils.loaders import sniff_and_load
from .deps import embedder, store, llm
from hashlib import md5
from typing import List

app = FastAPI(title="RAG Backend (Groq + Pinecone)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), namespace: str = Form("default"), doc_id: str = Form(None)):
    try:
        content = await file.read()
        text, kind = sniff_and_load(file.filename, content)
        if not text.strip():
            raise HTTPException(400, "No extractable text")

        chunks = chunk_text(text, chunk_size=800, overlap=120)
        embeds = embedder.embed(chunks)

        if not doc_id:
            # stable doc id
            doc_id = md5((file.filename + str(len(chunks))).encode()).hexdigest()[:12]

        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeds)):
            vec_id = f"{doc_id}-{i}"
            meta = {"text": chunk, "doc_id": doc_id, "source": file.filename, "type": kind, "i": i}
            vectors.append((vec_id, emb, meta))

        store.upsert(vectors=vectors, namespace=namespace)
        return {"ok": True, "doc_id": doc_id, "chunks": len(chunks), "namespace": namespace}
    except Exception as e:
        raise HTTPException(500, f"Ingest error: {e}")

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        q_emb = embedder.embed_one(req.question)
        res = store.query(q_emb, top_k=req.top_k, namespace=req.namespace)
        matches = res.get("matches", []) if isinstance(res, dict) else res.matches  # client may return object/dict

        # Build context
        ctx_parts: List[str] = []
        for m in matches:
            md = m.get("metadata") if isinstance(m, dict) else m.metadata
            if md and md.get("text"):
                ctx_parts.append(md["text"])
        context = "\n---\n".join(ctx_parts)

        answer = llm.answer(req.question, context)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(500, f"Query error: {e}")