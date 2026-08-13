import logfire

from app.agents.state import AgentState
from app.services.retrieval.ranking_service import rerank_documents
from app.services.retrieval.vectordb_service import search_enterprise_knowledge


def retrieve_node(state: AgentState):
    """
    Performs entity-aware vector search (Top-40 candidates) and semantic reranking (Top-8).
    """
    query = state["current_query"]
    entities = state.get("extracted_entities", {})

    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Vector DB for: {query} with entities: {entities}")
        raw_results = search_enterprise_knowledge(query, filter_dict=entities, limit=40)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        doc_contents = [doc["content"] for doc in raw_results if doc.get("content")]

        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=8)
            logfire.info(f"Reranking complete. Selected top {len(reranked_contents)} most relevant chunks.")

        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]

    return {
        "documents": formatted_docs,
        "status": "Found technical context.",
        "plan": state["plan"] + [f"Context Retrieved ({len(formatted_docs)} chunks)"],
    }
