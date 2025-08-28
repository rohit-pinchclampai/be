import os
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

INDEX_NAME = "rag-bot"
DIMENSION = 1024   # voyage-3-large returns 1536
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"
