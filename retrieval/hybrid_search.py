"""
Hybrid retrieval: combines BM25 (keyword/sparse) search with pgvector
(embedding/dense) search using Reciprocal Rank Fusion (RRF).

Why hybrid instead of just embeddings:
Dense embeddings are great at "meaning" (e.g. "session handling" ~ Session
class) but can miss exact-match cases that matter a lot in code — searching
for an exact function name, an error code, or a specific variable is often
better served by keyword search. BM25 catches those; embeddings catch
everything else. RRF combines both rankings without needing to tune a
weighting factor between them.

Usage:
    python retrieval/hybrid_search.py "how does session handling work"
"""
import os
import sys
import re
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import psycopg2

DB_CONFIG = {
    "host":os.environ.get("DB_HOST", "localhost"),
    "port": 5432,
    "dbname": "lumosai",
    "user": "postgres",
    "password": "lumos123",
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
RRF_K = 60  # standard RRF damping constant, rarely needs tuning


@dataclass
class RetrievedChunk:
    id: int
    file_path: str
    symbol_name: str | None
    symbol_type: str | None
    start_line: int
    end_line: int
    content: str
    rrf_score: float

    def citation(self) -> str:
        symbol = f" ({self.symbol_type} {self.symbol_name})" if self.symbol_name else ""
        return f"{self.file_path}:{self.start_line}-{self.end_line}{symbol}"


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25: lowercase, split on non-alphanumeric,
    but keep underscores so identifiers like `hash_password` stay intact
    as one meaningful token rather than splitting into `hash` + `password`.
    """
    text = text.lower()
    return re.findall(r"[a-z0-9_]+", text)


class HybridRetriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.conn = psycopg2.connect(**DB_CONFIG)
        self._load_bm25_index()

    def _load_bm25_index(self):
        """Pull all chunks into memory once and build a BM25 index over
        them. For a repo-sized dataset (thousands of chunks) this is fast
        and memory-cheap; if this ever needs to scale to huge monorepos,
        this is the piece you'd swap for Elasticsearch instead.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id, file_path, symbol_name, symbol_type, start_line, end_line, content FROM chunks;")
        rows = cur.fetchall()
        cur.close()

        self.row_by_id = {r[0]: r for r in rows}
        self.ids_in_order = [r[0] for r in rows]
        tokenized_corpus = [_tokenize(r[6]) for r in rows]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _bm25_ranked_ids(self, query: str, top_k: int) -> list[int]:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.ids_in_order, scores), key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, score in ranked[:top_k]]

    def _vector_ranked_ids(self, query: str, top_k: int) -> list[int]:
        query_embedding = self.model.encode(query).tolist()
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id FROM chunks
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
            """,
            (query_embedding, top_k),
        )
        ids = [row[0] for row in cur.fetchall()]
        cur.close()
        return ids

    def search(self, query: str, top_k: int = TOP_K, candidate_pool: int = 20) -> list[RetrievedChunk]:
        """Run both retrieval methods, fuse rankings with RRF, return top_k.

        candidate_pool: how many results to pull from EACH method before
        fusion — wider than top_k so fusion has enough overlap to work with.
        """
        bm25_ids = self._bm25_ranked_ids(query, candidate_pool)
        vector_ids = self._vector_ranked_ids(query, candidate_pool)

        # Reciprocal Rank Fusion: each id gets 1/(RRF_K + rank) from each
        # list it appears in; scores from both lists are summed. An id that
        # ranks well in BOTH lists rises to the top — this is the whole
        # point of hybrid search.
        rrf_scores: dict[int, float] = {}
        for rank, chunk_id in enumerate(bm25_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, chunk_id in enumerate(vector_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

        ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for chunk_id, score in ranked_ids:
            row = self.row_by_id[chunk_id]
            results.append(
                RetrievedChunk(
                    id=row[0],
                    file_path=row[1],
                    symbol_name=row[2],
                    symbol_type=row[3],
                    start_line=row[4],
                    end_line=row[5],
                    content=row[6],
                    rrf_score=score,
                )
            )
        return results

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "how does session handling work"

    retriever = HybridRetriever()
    results = retriever.search(query)

    print(f'\nQuery: "{query}"\n')
    print(f"Top {len(results)} hybrid results:\n")
    for r in results:
        print(f"  [RRF {r.rrf_score:.4f}] {r.citation()}")

    retriever.close()