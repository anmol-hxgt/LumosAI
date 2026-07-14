"""
Clones a GitHub repo and re-indexes it, replacing whatever was previously
indexed. This powers the "paste a GitHub URL" feature in the UI.

Deliberately single-repo scope: each call WIPES the existing index and
replaces it. Supporting multiple simultaneously-indexed repos with a
switcher is a reasonable future extension, not built here.
"""

import os
import shutil
import subprocess
import re

from ingestion.chunker import chunk_repository
from retrieval.embed_and_store import embed_and_store

CLONE_DIR = os.path.join(os.path.dirname(__file__), "indexed_repo")


def _validate_github_url(url: str) -> bool:
    return bool(re.match(r"^https://github\.com/[\w\-\.]+/[\w\-\.]+/?$", url.strip()))


def _extract_repo_name(url: str) -> str:
    return url.rstrip("/").split("/")[-1].replace(".git", "")


def clone_and_index(repo_url: str) -> dict:
    """Clones the given GitHub repo URL and re-indexes it. Returns a summary
    dict with counts, or raises an exception with a clear message on failure.
    """
    repo_url = repo_url.strip()
    if not _validate_github_url(repo_url):
        raise ValueError(
            "That doesn't look like a valid GitHub repo URL. "
            "Expected format: https://github.com/owner/repo"
        )

    # Clean up any previous clone before cloning fresh
    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR, ignore_errors=True)

    # Shallow clone (--depth 1) — we only need the current file contents,
    # not the full commit history, which makes this much faster.
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, CLONE_DIR],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

    # Quick sanity check before we spend time chunking/embedding
    chunks = chunk_repository(CLONE_DIR)
    if len(chunks) == 0:
        raise RuntimeError(
            "No supported code or doc files found in this repo — nothing to index."
        )

    # embed_and_store() already handles: chunking, embedding, TRUNCATE + insert.
    # We call chunk_repository above just to fail fast with a clear error
    # before committing to the (slower) embedding step.
    embed_and_store(CLONE_DIR)

    return {
        "repo_name": _extract_repo_name(repo_url),
        "total_chunks": len(chunks),
        "total_files": len(set(c.metadata.file_path for c in chunks)),
    }