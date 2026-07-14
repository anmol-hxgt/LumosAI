"""
Week 3, Step 2: Agent with tool-calling.

Extends basic_rag.py by giving the LLM access to tools (starting with just
the calculator). The core addition here is the LOOP: the LLM can request a
tool call, we execute it, feed the result back, and let the LLM decide if
it needs to call another tool or is ready to answer.

Usage:
    python agent/agent_with_tools.py "how does session handling work"
    python agent/agent_with_tools.py "if the timeout is 30 seconds and I retry 4 times, what's the max total wait?"
"""

import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from groq import Groq

from retrieval.hybrid_search import HybridRetriever
from agent.tools import TOOLS_SCHEMA, TOOL_FUNCTIONS

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"  # was llama-3.3-70b-versatile
MAX_TOOL_ITERATIONS = 5  # safety cap so a confused model can't loop forever

SYSTEM_PROMPT = """You are LumosAI, a codebase Q&A assistant. Answer the user's \
question using the provided code context below. Always cite the specific file \
and line numbers you used, like `sessions.py:511-555`. If the context doesn't \
contain enough information, say so explicitly instead of guessing. \
You have access to a calculator tool — use it for any arithmetic instead of \
computing it yourself, since you are prone to arithmetic mistakes."""


def build_context_block(chunks) -> str:
    parts = []
    for chunk in chunks:
        parts.append(f"--- {chunk.citation()} ---\n{chunk.content}\n")
    return "\n".join(parts)


def answer_question(question: str, top_k: int = 5) -> str:
    retriever = HybridRetriever()
    chunks = retriever.search(question, top_k=top_k)
    retriever.close()

    context = build_context_block(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context from the codebase:\n\n{context}\n\nQuestion: {question}",
        },
    ]

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",  # let the model decide whether a tool is needed
            temperature=0.2,
        )

        message = response.choices[0].message

        # No tool call requested — the model is ready to give a final answer.
        if not message.tool_calls:
            return message.content

        # The model wants to call one or more tools. We must append its
        # request to the conversation, then append each tool's result,
        # before asking it to continue.
        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"  [tool call] {tool_name}({tool_args})")

            tool_function = TOOL_FUNCTIONS.get(tool_name)
            if tool_function is None:
                result = f"Error: unknown tool '{tool_name}'"
            else:
                result = tool_function(**tool_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    return "Reached max tool iterations without a final answer — something may be looping."


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how does session handling work"
    print(f'\nQuestion: "{question}"\n')
    answer = answer_question(question)
    print(f"\n{answer}")