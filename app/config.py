from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PORT: int = 8000

    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "rag-index"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_METRIC: str = "cosine"

    EMBED_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBED_DIM: int = 768

    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    class Config:
        env_file = ".env"

settings = Settings()
