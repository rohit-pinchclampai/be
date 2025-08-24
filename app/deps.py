from .config import settings
from .services.embedder import Embedder
from .services.store import Store
from .services.groq_llm import GroqLLM

# Singletons (imported by main.py)
embedder = Embedder(settings.EMBED_MODEL_NAME)
store = Store(
    api_key=settings.PINECONE_API_KEY,
    index_name=settings.PINECONE_INDEX,
    dim=settings.EMBED_DIM,
    cloud=settings.PINECONE_CLOUD,
    region=settings.PINECONE_REGION,
    metric=settings.PINECONE_METRIC,
)
llm = GroqLLM(settings.GROQ_API_KEY, settings.GROQ_MODEL)
