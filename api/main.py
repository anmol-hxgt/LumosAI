"""
Week 5: FastAPI app with streaming, dashboard stats, and file listing.
"""

import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import psycopg2

from agent.agent_with_tools import answer_question
from retrieval.hybrid_search import HybridRetriever, DB_CONFIG
from agent.tools import TOOLS_SCHEMA, TOOL_FUNCTIONS
from repo_indexer import clone_and_index
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LumosAI", description="AI-Powered Code Intelligence Platform")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

# Stronger, more explicit priority ordering: codebase context comes FIRST,
# tools are a last resort — not an equal option the model picks freely.
# This directly addresses the "called web_search for a question retrieval
# already answered" behavior observed during testing.
SYSTEM_PROMPT = """You are LumosAI, a codebase Q&A assistant. Answer the user's \
question using the provided code context below whenever it contains relevant \
information — even if the context is only partially complete, prefer \
synthesizing an answer from it over reaching for a tool. \
Always cite the specific file and line numbers you used, like `sessions.py:511-555`. \

Only use tools when the codebase context genuinely cannot answer the question. \
Do NOT call any tool just to "think through" or double check something trivial \
— tools cost real time and money, only call one when its result will directly \
change your answer:
- Use web_search ONLY for information that cannot exist in this codebase \
(e.g. current library versions on PyPI, known CVEs, external documentation) — \
NEVER use it for questions about how THIS codebase's own code works, even if \
the context seems incomplete.
- Use calculator ONLY when the user's question itself requires a numeric \
calculation — never invent unrelated math to "verify" a conceptual answer.
- Use code_execution to verify behavior that requires actually running code.

If the context doesn't contain enough information and no tool applies, say so \
explicitly instead of guessing."""


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class IndexRepoRequest(BaseModel):
    repo_url: str


@app.get("/")
def serve_chat_ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/health")
def health_check():
    return {"status": "LumosAI is running"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    answer = answer_question(request.question)
    return QueryResponse(answer=answer)


@app.post("/api/index-repo")
def index_repo(request: IndexRepoRequest):
    """Clones and indexes a GitHub repo, replacing whatever was previously
    indexed. This is a blocking call — cloning + chunking + embedding a
    real repo can take anywhere from ~30 seconds to several minutes
    depending on repo size, which the frontend surfaces as a loading state.
    """
    try:
        result = clone_and_index(request.repo_url)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Dashboard endpoints — power the sidebar stats + file list in the UI
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def get_stats():
    """Returns high-level counts for the dashboard: total chunks, total
    unique files, and a breakdown by language.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM chunks;")
    total_chunks = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT file_path) FROM chunks;")
    total_files = cur.fetchone()[0]

    cur.execute(
        "SELECT language, COUNT(*) FROM chunks GROUP BY language ORDER BY COUNT(*) DESC;"
    )
    language_breakdown = [{"language": lang, "count": count} for lang, count in cur.fetchall()]

    cur.execute(
        "SELECT symbol_type, COUNT(*) FROM chunks GROUP BY symbol_type ORDER BY COUNT(*) DESC;"
    )
    symbol_type_breakdown = [{"type": t, "count": count} for t, count in cur.fetchall()]

    cur.close()
    conn.close()

    return {
        "total_chunks": total_chunks,
        "total_files": total_files,
        "language_breakdown": language_breakdown,
        "symbol_type_breakdown": symbol_type_breakdown,
    }


@app.get("/api/files")
def get_files():
    """Returns the list of indexed files with a chunk count for each,
    sorted alphabetically — powers the file list in the sidebar.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT file_path, language, COUNT(*) as chunk_count
        FROM chunks
        GROUP BY file_path, language
        ORDER BY file_path;
        """
    )
    files = [
        {"file_path": path, "language": lang, "chunk_count": count}
        for path, lang, count in cur.fetchall()
    ]

    cur.close()
    conn.close()

    return {"files": files}


@app.get("/api/files/chunks")
def get_file_chunks(file_path: str):
    """Returns every chunk belonging to a specific file, ordered by line
    number — this is what powers the source viewer when a user clicks a
    file in the sidebar.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT symbol_name, symbol_type, parent_symbol, start_line, end_line, content
        FROM chunks
        WHERE file_path = %s
        ORDER BY start_line;
        """,
        (file_path,),
    )
    chunks = [
        {
            "symbol_name": name,
            "symbol_type": stype,
            "parent_symbol": parent,
            "start_line": start,
            "end_line": end,
            "content": content,
        }
        for name, stype, parent, start, end, content in cur.fetchall()
    ]

    cur.close()
    conn.close()

    return {"file_path": file_path, "chunks": chunks}


# ---------------------------------------------------------------------------
# Streaming query
# ---------------------------------------------------------------------------

def build_context_block(chunks) -> str:
    parts = []
    for chunk in chunks:
        parts.append(f"--- {chunk.citation()} ---\n{chunk.content}\n")
    return "\n".join(parts)


def stream_answer(question: str):
    retriever = HybridRetriever()
    chunks = retriever.search(question, top_k=5)
    retriever.close()

    # Send the list of source files actually retrieved for THIS question,
    # before any tokens — lets the UI show "Sources: sessions.py, models.py"
    # immediately, distinct from the full indexed-file list in the sidebar.
    unique_sources = []
    seen = set()
    for c in chunks:
        if c.file_path not in seen:
            seen.add(c.file_path)
            unique_sources.append(c.file_path)
    yield f"data: {json.dumps({'type': 'sources', 'files': unique_sources})}\n\n"

    context = build_context_block(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context from the codebase:\n\n{context}\n\nQuestion: {question}"},
    ]

    try:
        initial_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.2,
        )
        message = initial_response.choices[0].message
    except Exception as e:
        # Smaller models occasionally hallucinate a malformed tool call
        # (e.g. inventing a "tool" that's actually a citation string like
        # 'sessions.py:395-905'), which Groq rejects with a 400 error.
        # Rather than letting this crash the whole SSE stream, retry once
        # WITHOUT tools, so the user still gets a real answer.
        yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'retry_without_tools', 'args': {'reason': str(e)[:150]}})}\n\n"
        fallback = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.2, stream=True
        )
        for chunk in fallback:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    if message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_function = TOOL_FUNCTIONS.get(tool_name)
            result = tool_function(**tool_args) if tool_function else f"Unknown tool: {tool_name}"

            yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': tool_args})}\n\n"

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
            )

        final_stream = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.2, stream=True
        )
        for chunk in final_stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
    else:
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.2, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    return StreamingResponse(stream_answer(request.question), media_type="text/event-stream")