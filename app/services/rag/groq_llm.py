import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

def groq_answer(question: str, context: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate an answer from Groq LLM given a context and question.
    API key is loaded from .env file.
    
    Args:
        question (str): The user's query.
        context (str): Retrieved context text from Pinecone or docs.
        model (str): The Groq LLM model name (default: mixtral-8x7b).

    Returns:
        str: Answer from the LLM.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("⚠️ GROQ_API_KEY not found in environment. Add it to your .env file.")

    client = Groq(api_key=api_key)

    system = (
        "You are a careful assistant. Answer ONLY from the provided context. "
        "If the answer isn't in the context, say you don't know."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content