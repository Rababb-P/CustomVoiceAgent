"""Agent tools. Retrieval is lazy-imported so the graph can be built and unit
tested without an index or embedding model on disk."""

from __future__ import annotations

from langchain_core.tools import tool

# Populated per turn by graph.py so the output guard can check groundedness
# against exactly what the model saw. Reset at the start of every turn.
LAST_RETRIEVED: list[str] = []


@tool
def search_life_info(query: str) -> str:
    """Search Rababb's personal corpus (resume, projects, work, education, FAQ)
    for facts relevant to the query. Always use this before answering factual
    questions about Rababb."""
    from src.rag.retrieve import retrieve

    chunks = retrieve(query)
    if not chunks:
        return "No relevant information found in the corpus for that query."
    LAST_RETRIEVED.extend(c.text for c in chunks)
    return "\n\n".join(f"[{c.cite()}]\n{c.text}" for c in chunks)


@tool
def list_topics() -> str:
    """List what's in Rababb's corpus (table of contents). Use this to decide
    whether a question is even answerable before searching."""
    from src.rag.retrieve import list_sources

    sources = list_sources()
    return "\n".join(sources) if sources else "Corpus is empty — nothing is knowable yet."


@tool
def clarify(question: str) -> str:
    """Ask the user a clarifying question instead of answering. Terminal: the
    question is spoken to the user and the turn ends."""
    return f"CLARIFY:{question}"


ALL_TOOLS = [search_life_info, list_topics, clarify]
