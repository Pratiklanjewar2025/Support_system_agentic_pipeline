"""
Knowledge Agent — retrieves relevant chunks from the FAISS vector store
and formats them as context for downstream agents.
"""

from agents.state import AgentState
from rag.retriever import retrieve


def knowledge_agent(state: AgentState) -> AgentState:
    query = state["user_query"]
    # top_k=2 and truncated chunks keep context small for free-tier rate limits
    results = retrieve(query, top_k=2)
    state["kb_results"] = results

    if results:
        context_parts = []
        for i, r in enumerate(results, 1):
            # Truncate each chunk to 400 chars to avoid oversized prompts
            snippet = r["text"][:400].strip()
            context_parts.append(
                f"[Source {i}: {r['source']}]\n{snippet}"
            )
        state["kb_context"] = "\n\n---\n\n".join(context_parts)
    else:
        state["kb_context"] = "No relevant documentation found."

    return state




