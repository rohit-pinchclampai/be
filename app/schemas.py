from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    namespace: str = "default"

class QueryResponse(BaseModel):
    answer: str
