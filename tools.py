"""
tools.py — available tools for the Task Execution Agent
========================================================
Each tool is a plain Python function.  Add your own by:
  1. Writing the function below
  2. Adding an entry to TOOLS

The executor node calls run_tool(name, input_str) automatically.
"""

from __future__ import annotations
import datetime
import json


# tool definitions 

TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for information on a topic",
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression, e.g. '12 * 7 + 3'",
    },
    {
        "name": "datetime",
        "description": "Get the current date and time (no input needed)",
    },
    {
        "name": "note",
        "description": "Save a short note or finding to the session log",
    },
]


# tool implementations 

_notes: list[str] = []   # in-memory note store


def _web_search(query: str) -> str:
    """
    Stub: replace with a real search API (e.g. Tavily, SerpAPI, Brave).
    Returns a mock result so the agent can run without API keys.
    """
    mock_results = {
        "python project structure": (
            "Best practices: use src/ layout, keep tests/ at root, "
            "pyproject.toml for metadata, .env for secrets, README.md."
        ),
        "default": (
            f"[Mock search result for '{query}'] "
            "Top result: Wikipedia article with overview of the topic."
        ),
    }
    key = next((k for k in mock_results if k in query.lower()), "default")
    return mock_results[key]


def _calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _datetime(_: str) -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _note(content: str) -> str:
    _notes.append(content)
    return f"Note saved ({len(_notes)} total)."


# dispatch

_REGISTRY: dict[str, callable] = {
    "web_search": _web_search,
    "calculator": _calculator,
    "datetime": _datetime,
    "note": _note,
}


def run_tool(name: str, input_str: str) -> str:
    fn = _REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: {name!r}. Available: {list(_REGISTRY)}"
    return fn(input_str)


def get_notes() -> list[str]:
    """Retrieve all saved notes after a run."""
    return list(_notes)
