"""
Agentic Router and Orchestrator.
Analyzes user intent, checks document relevance against score thresholds,
and dynamically routes questions to RAG, Web Search, or General LLM.
"""
from __future__ import annotations
import re
import logging
from typing import List, Dict, Any, Optional, Generator

from rag import retriever, reranker, generator, web_search
from rag.config import (
    DEFAULT_DENSE_TOP_K,
    DEFAULT_SPARSE_TOP_K,
    DEFAULT_TOP_N_RERANK,
    DEFAULT_LLM_TOP_K,
)

logger = logging.getLogger(__name__)

# Minimum cross-encoder threshold for document relevance
RELEVANCE_THRESHOLD = -2.5


def is_follow_up(query: str) -> bool:
    """Detect follow-up queries that reference previous turns."""
    q = query.lower().strip()
    follow_up_phrases = [
        "explain it", "explain that", "tell me more", "what about that", "elaborate",
        "more simply", "simpler", "what is its", "what are its", "why is that",
        "how does that work", "can you explain further", "give an example",
    ]
    if any(phrase in q for phrase in follow_up_phrases):
        return True
    words = q.split()
    if len(words) <= 5 and any(pronoun in words for pronoun in ["it", "its", "that", "this", "them", "these"]):
        return True
    return False


def resolve_follow_up(query: str, conversation_history: Optional[List[Dict[str, str]]]) -> str:
    """Enrich follow-up query with subject context from previous conversation turns."""
    if not conversation_history:
        return query

    # Locate the last user query
    last_user_turn = ""
    for msg in reversed(conversation_history):
        if msg.get("role") == "user":
            last_user_turn = msg.get("content", "")
            break

    if not last_user_turn:
        return query

    q_lower = query.lower()
    # Check for specific simplification request
    if "more simply" in q_lower or "simpler" in q_lower or "simply" in q_lower:
        return f"Explain {last_user_turn} in very simple, beginner-friendly terms with examples."

    # Extract nouns/keywords from previous turn to resolve "its / it / that"
    cleaned_last = re.sub(r"[^\w\s]", "", last_user_turn)
    keywords = [w for w in cleaned_last.split() if len(w) > 3 and w.lower() not in ["what", "explain", "about", "tell", "this", "that"]]
    subject = " ".join(keywords) if keywords else last_user_turn

    return f"{query} (Context: {subject})"


def is_document_specific_query(query: str) -> bool:
    """Detect queries explicitly referring to an uploaded document."""
    q_lower = query.lower()
    return bool(re.search(r"\b(pdf|document|file|handbook|chapter|section|page|uploaded)\b", q_lower))


def run_agentic_rag(
    query: str,
    document_ids: List[str],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    dense_top_k: int = DEFAULT_DENSE_TOP_K,
    sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
    top_n_rerank: int = DEFAULT_TOP_N_RERANK,
    llm_top_k: int = DEFAULT_LLM_TOP_K,
    provider: str = "extractive",
    model_name: str = "google/flan-t5-base",
    temperature: float = 0.3,
    use_cross_encoder: bool = True,
    watsonx_config: Optional[Dict[str, str]] = None,
    openai_config: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Intelligent routing and execution:
    1. Check for live/real-time web queries -> Web Search.
    2. If documents are loaded -> Hybrid Retrieval + CrossEncoder Rerank.
    3. Evaluate relevance threshold:
       - If relevant -> RAG Answer (+ View Sources).
       - If not relevant -> Fallback to Web Search or General LLM.
    4. If no documents loaded -> Web Search for live info, or General LLM.
    """
    has_documents = len(document_ids) > 0
    resolved_query = resolve_follow_up(query, conversation_history) if is_follow_up(query) else query

    logger.info(f"Agent analyzing: '{query}' -> Resolved: '{resolved_query}' | Docs available: {has_documents}")

    # -------------------------------------------------------------
    # CASE 1: Live / Real-Time Time-Sensitive Query
    # (e.g. stock market today, AI news today, who won today's match)
    # -------------------------------------------------------------
    if web_search.is_live_or_web_query(resolved_query) and not is_document_specific_query(resolved_query):
        logger.info("Routing to Web Search: query detected as time-sensitive/current.")
        web_results = web_search.search_web(resolved_query, max_results=4)
        return {
            "route": "web",
            "intent": "WEB_SEARCH",
            "stream_generator": generator.generate_web_response(resolved_query, web_results),
            "retrieved_sources": [],
            "web_sources": web_results,
            "stats": {"route": "web", "web_results_count": len(web_results)},
        }

    # -------------------------------------------------------------
    # CASE 2: No Documents Uploaded
    # -------------------------------------------------------------
    if not has_documents:
        # If user explicitly asked for document content
        if is_document_specific_query(resolved_query):
            def no_doc_gen():
                yield "📁 No document is currently uploaded. Please upload your PDF, DOCX, or text file in the sidebar to ask questions about it! 📚"

            return {
                "route": "direct",
                "intent": "NO_DOCUMENT_PROMPT",
                "stream_generator": no_doc_gen(),
                "retrieved_sources": [],
                "web_sources": [],
                "stats": {"route": "direct"},
            }

        # Check if general knowledge or general web search
        logger.info("No documents uploaded: attempting web search / general answer.")
        web_results = web_search.search_web(resolved_query, max_results=3)
        if web_results:
            return {
                "route": "web",
                "intent": "WEB_SEARCH",
                "stream_generator": generator.generate_web_response(resolved_query, web_results),
                "retrieved_sources": [],
                "web_sources": web_results,
                "stats": {"route": "web", "count": len(web_results)},
            }
        else:
            return {
                "route": "direct",
                "intent": "GENERAL_KNOWLEDGE",
                "stream_generator": generator.generate_general_knowledge(resolved_query),
                "retrieved_sources": [],
                "web_sources": [],
                "stats": {"route": "direct"},
            }

    # -------------------------------------------------------------
    # CASE 3: Documents Are Uploaded -> Hybrid Retrieval & Relevance Check
    # -------------------------------------------------------------
    hybrid_out = retriever.hybrid_search(
        query=resolved_query,
        document_ids=document_ids,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        top_k=dense_top_k + sparse_top_k,
    )
    candidates = hybrid_out["results"]

    ranked_chunks = reranker.rerank(
        query=resolved_query,
        chunks=candidates,
        top_n=top_n_rerank,
        use_cross_encoder=use_cross_encoder,
    )

    # Determine Relevance
    top_score = ranked_chunks[0].get("rerank_score", -999.0) if ranked_chunks else -999.0
    logger.info(f"Top rerank score: {top_score} (Threshold: {RELEVANCE_THRESHOLD})")

    # Keyword presence check for query terms in top chunks
    query_words = set(re.findall(r"\w+", resolved_query.lower())) - {"what", "is", "the", "how", "why", "explain", "in", "of", "and", "a", "an"}
    top_texts = " ".join([c.get("text", "").lower() for c in ranked_chunks[:3]])
    has_keyword_match = any(qw in top_texts for qw in query_words)

    is_relevant = (top_score >= RELEVANCE_THRESHOLD) or has_keyword_match or is_document_specific_query(resolved_query)

    # -------------------------------------------------------------
    # Sub-case 3A: Content IS Relevant to Uploaded Documents -> RAG
    # -------------------------------------------------------------
    if is_relevant and ranked_chunks:
        context_chunks = ranked_chunks[:llm_top_k]

        # Package sources for optional "🔎 View Sources" expander
        sources = []
        for c in ranked_chunks:
            meta = c.get("metadata", {})
            sources.append({
                "rank": c.get("rank", 1),
                "score": c.get("rerank_score", c.get("score", 0.0)),
                "source": meta.get("source", "Document"),
                "document_id": meta.get("document_id", ""),
                "page": meta.get("page_number", "N/A"),
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text", ""),
            })

        # Generate natural answer
        if provider == "watsonx":
            stream = generator.generate_watsonx(
                query=resolved_query,
                context_chunks=context_chunks,
                conversation_history=conversation_history,
                api_key=watsonx_config.get("api_key", "") if watsonx_config else "",
                url=watsonx_config.get("url", "") if watsonx_config else "",
                project_id=watsonx_config.get("project_id", "") if watsonx_config else "",
                model_id=watsonx_config.get("model_id", "") if watsonx_config else "",
                temperature=temperature,
            )
        elif provider == "openai":
            stream = generator.generate_openai_compatible(
                query=resolved_query,
                context_chunks=context_chunks,
                conversation_history=conversation_history,
                api_key=openai_config.get("api_key", "") if openai_config else "",
                base_url=openai_config.get("base_url", "") if openai_config else "",
                model_name=openai_config.get("model_name", "") if openai_config else "",
                temperature=temperature,
            )
        elif provider == "local":
            stream = generator.generate_local(
                query=resolved_query,
                context_chunks=context_chunks,
                conversation_history=conversation_history,
                model_name=model_name,
                temperature=temperature,
            )
        else:  # Natural Extractive fallback
            stream = generator.generate_natural_extractive(
                query=resolved_query,
                context_chunks=context_chunks,
            )

        return {
            "route": "rag",
            "intent": "DOCUMENT_RAG",
            "stream_generator": stream,
            "retrieved_sources": sources,
            "web_sources": [],
            "stats": {"route": "rag", "top_score": top_score, "chunks_used": len(context_chunks)},
        }

    # -------------------------------------------------------------
    # Sub-case 3B: Content NOT in Documents -> Route to Web Search
    # -------------------------------------------------------------
    logger.info(f"Query '{query}' not found in documents (score {top_score}). Routing to Web Search.")
    web_results = web_search.search_web(resolved_query, max_results=3)

    if web_results:
        def prefixed_web_gen():
            yield "ℹ️ *This topic was not found in your uploaded documents. Here is what I found on the web:*\n\n"
            yield from generator.generate_web_response(resolved_query, web_results)

        return {
            "route": "web",
            "intent": "WEB_FALLBACK",
            "stream_generator": prefixed_web_gen(),
            "retrieved_sources": [],
            "web_sources": web_results,
            "stats": {"route": "web_fallback", "top_doc_score": top_score},
        }

    # Neither document nor web search has sufficient info
    def not_found_gen():
        yield "I couldn't find information regarding that in your uploaded documents or from web sources. 📚🌐"

    return {
        "route": "direct",
        "intent": "NOT_FOUND",
        "stream_generator": not_found_gen(),
        "retrieved_sources": [],
        "web_sources": [],
        "stats": {"route": "not_found"},
    }
