"""
Week 3, Step 1: Basic RAG agent — no tools yet.

Takes a user question, retrieves relevant chunks using the hybrid retriever
from Week 2, and asks the LLM to answer USING ONLY that retrieved context.
This is the foundation the tool-calling agent (Step 2) will build on.

Usage:
    python agent/basic_rag.py "how does session handling work"
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from groq import Groq

from retrieval.hybrid_search import HybridRetriever

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"  # was llama-3.3-70b-versatile

SYSTEM_PROMPT = """You are LumosAI, a codebase Q&A assistant. Answer the user's \
question using ONLY the provided code context below. Always cite the specific \
file and line numbers you used in your answer, like `sessions.py:511-555`. \
If the context doesn't contain enough information to answer confidently, say \
so explicitly instead of guessing."""


def build_context_block(chunks) -> str:
    """Format retrieved chunks into a single context string for the prompt,
    each one clearly labeled with its citation so the model can reference it.
    """
    parts = []
    for chunk in chunks:
        parts.append(f"--- {chunk.citation()} ---\n{chunk.content}\n")
    return "\n".join(parts)


def answer_question(question: str, top_k: int = 5) -> str:
    retriever = HybridRetriever()
    chunks = retriever.search(question, top_k=top_k)
    retriever.close()

    context = build_context_block(chunks)

    user_message = f"""Context from the codebase:

{context}

Question: {question}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,  # low temperature — we want grounded, consistent answers, not creative ones
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how does session handling work"
    print(f'\nQuestion: "{question}"\n')
    print("Retrieving context and generating answer...\n")
    answer = answer_question(question)
    print(answer)