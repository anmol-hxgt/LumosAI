"""
Quick sanity check: embed a natural-language question, and find the closest
stored chunks using pgvector's cosine distance operator.

This is NOT the final retrieval module (that comes next, with BM25 fusion) —
just a fast way to confirm the embeddings we stored are actually useful
before we build more on top of them.

Usage:
    python retrieval/test_search.py "how does session handling work"
"""

import sys
from sentence_transformers import SentenceTransformer
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "lumosai",
    "user": "postgres",
    "password": "lumos123",
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def search(query: str, top_k: int = TOP_K):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode(query).tolist()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # <-> is pgvector's L2 distance operator; smaller = more similar.
    # (cosine distance <=> also works if you prefer that metric)
    cur.execute(
        """
        SELECT file_path, symbol_name, symbol_type, start_line, end_line,
               embedding <-> %s::vector AS distance
        FROM chunks
        ORDER BY distance
        LIMIT %s;
        """,
        (query_embedding, top_k),
    )
    results = cur.fetchall()

    print(f'\nQuery: "{query}"\n')
    print(f"Top {top_k} results:\n")
    for file_path, symbol_name, symbol_type, start_line, end_line, distance in results:
        print(f"  [{distance:.4f}] {file_path}:{start_line}-{end_line} ({symbol_type} {symbol_name})")

    cur.close()
    conn.close()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "how does session handling work"
    search(query)