# LumosAI — AI-Powered Code Intelligence Platform

A RAG-based codebase Q&A agent that lets you point at any GitHub repository and ask natural-language questions about it — grounded in the actual code, with citations, hybrid retrieval, agentic tool-use, and a rigorous RAGAS-based evaluation.

**Live demo:** [http://13.61.147.16:8000](http://13.61.147.16:8000) *(personal EC2 instance — may not always be running; see "Run it yourself" below if it's offline)*

---

## The problem

Developers joining a new codebase spend disproportionate time answering questions like "where is auth handled" or "how does the retry logic work" — reading through files and grepping doesn't scale in large repos where logic is split across many files. LumosAI answers these questions directly, citing the exact file and line numbers it used, and can reach for tools (web search, calculator, code execution) when retrieval alone isn't enough.

---

## Architecture

**Offline — ingestion & indexing:**
```
GitHub repository → AST-aware chunker (tree-sitter) → Dual index (BM25 + embeddings) → pgvector
```

**Runtime — query flow:**
```
User query → Hybrid retriever (BM25 + vector, RRF fusion) → Agent orchestrator (LLM + tool routing) → Streaming response (FastAPI + SSE)
```

The agent can call three tools when retrieval alone can't answer: **web_search** (for info outside the indexed repo, e.g. current library versions), **calculator** (AST-restricted, not `eval()`), and **code_execution** (subprocess-isolated with a timeout).

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Chunking | `tree-sitter` (AST-aware) | Splits by function/class/method boundaries, not fixed character windows — a function is never cut mid-body |
| Sparse retrieval | BM25 (`rank_bm25`) | Exact-match strength for identifiers/function names |
| Dense retrieval | `sentence-transformers` (`all-MiniLM-L6-v2`) | Semantic/conceptual matching |
| Fusion | Reciprocal Rank Fusion | Combines both rankings without needing a tuned weighting factor |
| Vector store | PostgreSQL + `pgvector` | Reuses a tool already in the stack instead of a separate vector DB |
| Agent loop | Hand-built (Groq API, OpenAI-compatible tool-calling) | Every step is explainable — no framework abstracting the decide → call → observe → answer loop |
| Evaluation | RAGAS (faithfulness, relevancy, context precision/recall) | Measures retrieval and generation quality against a hand-labeled eval set, not just a demo |
| Serving | FastAPI + Server-Sent Events | Token-by-token streaming responses |
| Frontend | Vanilla HTML/JS | Dashboard sidebar (file/chunk stats, clickable source viewer), no build step |
| Deployment | Docker Compose on AWS EC2 | App + Postgres containers, same pattern as production systems |

---

## Features

- **Live repo indexing** — paste any GitHub URL in the UI; it clones, chunks, embeds, and indexes it, replacing the previous index
- **Source viewer** — click any indexed file to see every chunk (function/class/method) extracted from it, with line numbers
- **Per-answer sources** — every response shows exactly which files were retrieved for that specific question, clickable to view
- **Tool-calling** — the agent decides when to use web search, calculator, or code execution, rather than always reaching for a tool
- **Streaming responses** — answers appear token-by-token, not all at once

---

## Evaluation results

Evaluated with RAGAS against a 15-question hand-labeled set spanning 4 difficulty categories (lookup, multi-hop, call-chain, architecture) on the `requests` library. Full writeup: [`EVAL_RESULTS.md`](./EVAL_RESULTS.md).

| Category | Faithfulness | Context Precision | Context Recall |
|---|---|---|---|
| Lookup | 1.00 | 0.95 | 1.00 |
| Multi-hop | 0.93 | 0.89 | 1.00 |
| Call-chain | 1.00 | 1.00 | 0.67 |
| Architecture | 0.56 | 1.00 | 1.00 |

**Key finding:** retrieval quality stays strong even on complex multi-hop and architecture-level questions. The weak point is generation, not retrieval — on architecture-synthesis questions, the model sometimes hedges rather than confidently synthesizing across retrieved chunks, which RAGAS correctly penalizes as lower faithfulness. This is a specific, testable direction for improvement rather than a vague "make it better."

*(Note: this run was constrained by free-tier API rate limits and reflects a partial but real sample — see EVAL_RESULTS.md for full methodology and honest limitations.)*

---

## Known limitations

- **RST doc chunking:** `.rst` files using underline-style headings (common in Sphinx docs) aren't detected by the heading-aware splitter and fall back to blind character-window chunking. Markdown docs and all code chunking are unaffected. (Confirmed while indexing Flask's docs — see `NOTES.md`.)
- **Code execution sandboxing:** the `code_execution` tool isolates via a subprocess + timeout, not a full Docker sandbox — no network/memory isolation. A production version would run this in a throwaway container per execution.
- **Smaller-model tool judgment:** running on `llama-3.1-8b-instant` (free tier) occasionally produces irrelevant tool calls or malformed tool-call syntax; the API layer catches this and retries without tools rather than failing the whole request.

---

## Run it yourself

```bash
git clone https://github.com/anmol-hxgt/LumosAI.git
cd LumosAI
cp .env.example .env   # add your GROQ_API_KEY and TAVILY_API_KEY
docker-compose up --build
```

Then open `http://localhost:8000`, paste a GitHub URL into the sidebar, and ask a question once indexing finishes.

---

## Project structure

```
lumos-ai/
├── ingestion/       # AST-aware chunker (tree-sitter)
├── retrieval/       # Hybrid BM25 + vector search, embedding/storage
├── agent/           # Tool-calling agent loop, tool implementations
├── api/             # FastAPI app, streaming endpoint, static UI
├── eval/            # RAGAS evaluation set and runner
├── repo_indexer.py  # Clone + index a GitHub repo on demand
├── docker-compose.yml
└── Dockerfile
```