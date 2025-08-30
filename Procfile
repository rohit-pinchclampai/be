web: gunicorn app.services.rag.app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120 --workers 2
