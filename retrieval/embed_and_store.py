"""
Embeds chunks produced by the Week-1 chunker and stores them in Postgres/pgvector.

Usage:
    python retrieval/embed_and_store.py <path_to_repo>

This clears out any existing rows for a fresh run (fine for now, during
development) and re-inserts everything from scratch.
"""

import sys
import os

# Allow importing ingestion/chunker.py when running this script directly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sentence_transformers import SentenceTransformer
import psycopg2

from ingestion.chunker import chunk_repository

# ---------------------------------------------------------------------------
# Config — adjust if your Docker container uses different credentials
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": 5432,
    "dbname": "lumosai",
    "user": "postgres",
    "password": "lumos123",
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32  # how many chunks to embed at once — keeps memory usage reasonable


def embed_and_store(repo_path: str):
    print(f"Chunking repository: {repo_path}")
    chunks = chunk_repository(repo_path)
    print(f"Got {len(chunks)} chunks. Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Clear existing rows so re-running this script during development
    # doesn't just keep appending duplicates.
    cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY;")
    conn.commit()

    print(f"Embedding and inserting {len(chunks)} chunks in batches of {BATCH_SIZE}...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.text for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False)

        for chunk, embedding in zip(batch, embeddings):
            m = chunk.metadata
            cur.execute(
                """
                INSERT INTO chunks
                    (file_path, symbol_name, symbol_type, parent_symbol,
                     start_line, end_line, language, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    m.file_path,
                    m.symbol_name,
                    m.symbol_type,
                    m.parent_symbol,
                    m.start_line,
                    m.end_line,
                    m.language,
                    chunk.text,
                    embedding.tolist(),  # pgvector accepts a plain list of floats
                ),
            )

        conn.commit()
        print(f"  Inserted {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    cur.execute("SELECT COUNT(*) FROM chunks;")
    total = cur.fetchone()[0]
    print(f"\nDone. {total} rows now in the chunks table.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "test_repo"
    embed_and_store(target)