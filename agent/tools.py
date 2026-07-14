"""
Tools the agent can call. Each tool is a plain Python function, plus a
matching JSON-schema description that gets sent to the LLM so it knows
the tool exists and what arguments it takes.

Tools: calculator, web_search, code_execution.
"""

import ast
import operator
import os
import subprocess
import sys
import tempfile

from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Calculator — safe expression evaluation (NOT using eval(), which would let
# arbitrary code run if the LLM ever passed something malicious/weird)
# ---------------------------------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression like '45 * 12 + 3'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ---------------------------------------------------------------------------
# Web search — for questions the indexed codebase can't answer
# ---------------------------------------------------------------------------

_tavily_client = None


def _get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set in environment/.env")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def web_search(query: str) -> str:
    """Search the web and return a short summary of the top results."""
    try:
        client = _get_tavily_client()
        response = client.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            return "No results found."

        formatted = []
        for r in results:
            formatted.append(f"- {r['title']}: {r['content'][:200]}... (source: {r['url']})")
        return "\n".join(formatted)
    except Exception as e:
        return f"Error performing web search: {e}"


# ---------------------------------------------------------------------------
# Code execution — runs Python in an isolated SUBPROCESS with a timeout.
#
# HONEST LIMITATION (say this out loud in interviews, don't hide it):
# This isolates via a separate OS process + timeout, which stops infinite
# loops and keeps a crash from taking down the main agent process. It does
# NOT provide true sandboxing — no network restriction, no memory cap, no
# filesystem isolation. A production version would run this inside a
# throwaway Docker container per execution (network disabled, memory/CPU
# limited, filesystem read-only except a scratch dir) instead of a bare
# subprocess. That's a known, deliberate scope cut for this project stage.
# ---------------------------------------------------------------------------

CODE_EXECUTION_TIMEOUT_SECONDS = 5


def code_execution(code: str) -> str:
    """Run a short Python snippet and return its stdout/stderr. Use this to
    verify behavior (e.g. does this regex match this string, what does this
    expression evaluate to) rather than reasoning about it purely in text.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=CODE_EXECUTION_TIMEOUT_SECONDS,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: code execution exceeded {CODE_EXECUTION_TIMEOUT_SECONDS}s timeout (possible infinite loop)"
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Tool schemas — sent to the LLM so it knows what tools exist
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a basic arithmetic expression. Use this for any math the user asks about, instead of computing it yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A basic arithmetic expression, e.g. '45 * 12 + 3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for current information NOT available in the codebase context — e.g. latest library versions, known CVEs, external documentation, or anything requiring up-to-date knowledge beyond the indexed repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_execution",
            "description": "Executes a short Python snippet and returns its output. Use this to verify behavior instead of reasoning about it purely in text — e.g. checking if a regex matches, or what an expression evaluates to.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to produce visible output.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]

# Maps tool name -> actual Python function to call when the LLM requests it
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "web_search": web_search,
    "code_execution": code_execution,
}


if __name__ == "__main__":
    print(calculator("45 * 12 + 3"))
    print(code_execution("import re\nprint(bool(re.match(r'^[a-z]+$', 'hello')))"))
    print(code_execution("while True: pass"))  # should time out safely