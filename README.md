## Local run
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill keys
uvicorn app.main:app --reload

## Test
curl -F "file=@/path/doc.pdf" -F "namespace=default" http://localhost:8000/ingest
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question":"What is this doc about?","top_k":4,"namespace":"default"}'

## Pinecone index
On first run the backend creates a serverless index with your chosen dimension/metric.

## Deploy
- Render: new Web Service -> Docker -> set env vars -> expose 8000
- Railway/Fly.io: similar; just provide Dockerfile and env vars
