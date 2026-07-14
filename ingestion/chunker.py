"""
AST-aware code chunker for the Codebase Q&A Agent.

Instead of splitting files by a fixed character/token window (which cuts
functions in half and destroys retrievability), this walks the AST and emits
one chunk per top-level function / class / method, with rich metadata
attached (file path, symbol name, line range, language, parent class if any).

Markdown / plain-text docs fall back to a heading-aware recursive splitter,
since prose doesn't have a meaningful AST for our purposes.

Usage:
    from chunker import chunk_repository

    chunks = chunk_repository("/path/to/repo")
    for c in chunks:
        print(c.metadata, len(c.text))
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from tree_sitter_languages import get_parser

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Map file extensions -> tree-sitter language name.
# Extend this as you add support for more languages.
LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
}

# Node types that represent a "unit worth its own chunk" per language.
# tree-sitter grammars name these differently per language, so this is the
# main thing you extend when adding a new language.
CHUNKABLE_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
    "tsx": {"function_declaration", "class_declaration", "method_definition"},
    "java": {"method_declaration", "class_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "cpp": {"function_definition", "class_specifier"},
    "c": {"function_definition"},
}

MAX_CODE_CHUNK_LINES = 80   # if a function is longer than this, sub-split it
DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".next"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    file_path: str
    symbol_name: str | None      # function/class name, or None for doc chunks
    symbol_type: str | None      # "function", "class", "method", "doc_section"
    parent_symbol: str | None    # enclosing class name, if this is a method
    start_line: int
    end_line: int
    language: str

    def as_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "parent_symbol": self.parent_symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
        }


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata

    def citation(self) -> str:
        """Human-readable citation, e.g. 'auth.py:45-67 (function login)'"""
        m = self.metadata
        symbol = f" ({m.symbol_type} {m.symbol_name})" if m.symbol_name else ""
        return f"{m.file_path}:{m.start_line}-{m.end_line}{symbol}"


# ---------------------------------------------------------------------------
# Code chunking (AST-based)
# ---------------------------------------------------------------------------

def _extract_signature_line(node_text: str) -> str:
    """Grab just the first line (the def/class signature) to prepend to
    sub-chunks of an oversized function, so retrieval still surfaces the
    function's identity even if the body got split."""
    first_line = node_text.splitlines()[0] if node_text else ""
    return first_line.strip()


def _split_oversized_chunk(text: str, signature: str, max_lines: int) -> list[str]:
    """If a function/class body is too long, split it into sub-chunks of
    at most `max_lines`, each prefixed with the original signature so the
    sub-chunk is still identifiable and searchable on its own."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return [text]

    sub_chunks = []
    for i in range(0, len(lines), max_lines):
        piece = "\n".join(lines[i : i + max_lines])
        if i > 0:
            piece = f"# ...continued from: {signature}\n{piece}"
        sub_chunks.append(piece)
    return sub_chunks


def _walk_ast_for_chunks(
    node, source_bytes: bytes, chunkable_types: set[str], parent_symbol: str | None = None
) -> Iterable[tuple]:
    """Recursively walk the tree-sitter AST, yielding (node, symbol_type,
    parent_symbol) for every node whose type is in `chunkable_types`.

    We recurse into class bodies to also pick up methods individually
    (so a huge class doesn't become one giant unsearchable chunk), while
    still emitting the class itself as a chunk for "what does this class do"
    style questions.
    """
    if node.type in chunkable_types:
        is_class = "class" in node.type
        symbol_type = "class" if is_class else ("method" if parent_symbol else "function")
        yield (node, symbol_type, parent_symbol)

        if is_class:
            # descend into the class body to also emit individual methods
            new_parent = _get_node_name(node, source_bytes)
            for child in node.children:
                yield from _walk_ast_for_chunks(child, source_bytes, chunkable_types, new_parent)
            return  # don't also descend generically below (avoid double-walk)

    for child in node.children:
        yield from _walk_ast_for_chunks(child, source_bytes, chunkable_types, parent_symbol)


def _get_node_name(node, source_bytes: bytes) -> str | None:
    """Pull the identifier name out of a function/class definition node."""
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
    return None


def chunk_code_file(file_path: Path, repo_root: Path) -> list[Chunk]:
    ext = file_path.suffix.lower()
    language = LANGUAGE_BY_EXTENSION.get(ext)
    if language is None:
        return []

    chunkable_types = CHUNKABLE_NODE_TYPES.get(language, set())
    if not chunkable_types:
        return []

    source_bytes = file_path.read_bytes()
    parser = get_parser(language)
    tree = parser.parse(source_bytes)

    relative_path = str(file_path.relative_to(repo_root))
    chunks: list[Chunk] = []

    for node, symbol_type, parent_symbol in _walk_ast_for_chunks(
        tree.root_node, source_bytes, chunkable_types
    ):
        node_text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        symbol_name = _get_node_name(node, source_bytes)
        start_line = node.start_point[0] + 1  # tree-sitter is 0-indexed
        end_line = node.end_point[0] + 1

        signature = _extract_signature_line(node_text)
        pieces = _split_oversized_chunk(node_text, signature, MAX_CODE_CHUNK_LINES)

        for piece in pieces:
            chunks.append(
                Chunk(
                    text=piece,
                    metadata=ChunkMetadata(
                        file_path=relative_path,
                        symbol_name=symbol_name,
                        symbol_type=symbol_type,
                        parent_symbol=parent_symbol,
                        start_line=start_line,
                        end_line=end_line,
                        language=language,
                    ),
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# Doc chunking (heading-aware, for markdown/rst/txt)
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def chunk_doc_file(file_path: Path, repo_root: Path) -> list[Chunk]:
    """Split markdown-like docs on heading boundaries so each chunk is one
    coherent section as the author intended, rather than an arbitrary
    character window. Falls back to fixed-size chunking with overlap for
    files with no headings at all (e.g. long unstructured .txt files).
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    relative_path = str(file_path.relative_to(repo_root))

    headings = list(_MD_HEADING_RE.finditer(text))
    if not headings:
        return _fallback_recursive_split(text, relative_path)

    chunks: list[Chunk] = []
    for i, match in enumerate(headings):
        section_start = match.start()
        section_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section_text = text[section_start:section_end].strip()
        if not section_text:
            continue

        start_line = text[:section_start].count("\n") + 1
        end_line = text[:section_end].count("\n") + 1
        heading_title = match.group(2).strip()

        chunks.append(
            Chunk(
                text=section_text,
                metadata=ChunkMetadata(
                    file_path=relative_path,
                    symbol_name=heading_title,
                    symbol_type="doc_section",
                    parent_symbol=None,
                    start_line=start_line,
                    end_line=end_line,
                    language="markdown",
                ),
            )
        )
    return chunks


def _fallback_recursive_split(
    text: str, relative_path: str, chunk_size: int = 800, overlap: int = 100
) -> list[Chunk]:
    """Simple overlapping character-window split, used only when a doc file
    has no heading structure to key off of at all."""
    chunks: list[Chunk] = []
    lines = text.splitlines()
    line_offsets = []
    pos = 0
    for line in lines:
        line_offsets.append(pos)
        pos += len(line) + 1

    step = max(chunk_size - overlap, 1)
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size]
        if not piece.strip():
            continue
        start_line = next((i for i, off in enumerate(line_offsets) if off >= start), 0) + 1
        end_line = start_line + piece.count("\n")
        chunks.append(
            Chunk(
                text=piece,
                metadata=ChunkMetadata(
                    file_path=relative_path,
                    symbol_name=None,
                    symbol_type="doc_fragment",
                    parent_symbol=None,
                    start_line=start_line,
                    end_line=end_line,
                    language="text",
                ),
            )
        )
        if start + chunk_size >= len(text):
            break
    return chunks


# ---------------------------------------------------------------------------
# Repository walker
# ---------------------------------------------------------------------------

def chunk_repository(repo_root: str | Path) -> list[Chunk]:
    """Walk a repository and return chunks for every supported code and doc
    file, skipping common noise directories (.git, node_modules, venvs, etc).
    """
    repo_root = Path(repo_root).resolve()
    all_chunks: list[Chunk] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

        for filename in filenames:
            file_path = Path(dirpath) / filename
            ext = file_path.suffix.lower()

            try:
                if ext in LANGUAGE_BY_EXTENSION:
                    all_chunks.extend(chunk_code_file(file_path, repo_root))
                elif ext in DOC_EXTENSIONS:
                    all_chunks.extend(chunk_doc_file(file_path, repo_root))
            except (UnicodeDecodeError, SyntaxError) as e:
                # Don't let one malformed file kill the whole ingestion run —
                # log and continue. In production, send this to a proper logger.
                print(f"[chunker] skipped {file_path} due to: {e}")
                continue

    return all_chunks


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    chunks = chunk_repository(target)

    print(f"Extracted {len(chunks)} chunks from {target}\n")
    for c in chunks[:10]:
        print(f"--- {c.citation()} ---")
        print(c.text[:200].strip())
        print()
