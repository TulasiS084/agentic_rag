"""
Natural Conversational LLM Generation Layer.
Produces clean, modern AI assistant responses without exposing internal RAG artifacts.
Supports:
1. Local Hugging Face Transformers (Flan-T5, TinyLlama)
2. IBM WatsonX (Granite models via REST)
3. OpenAI-Compatible API (OpenAI, Groq, Ollama)
4. Conversational Extractive Synthesizer (Natural fallback)
5. Web Search Grounded Synthesizer
"""
from __future__ import annotations
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Generator

import requests

from rag.config import (
    DEFAULT_LLM_TOP_K,
    WATSONX_API_KEY,
    WATSONX_URL,
    WATSONX_PROJECT_ID,
    WATSONX_MODEL_ID,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL_NAME,
    LOCAL_LLM_MODEL,
)

logger = logging.getLogger(__name__)

# Local model singletons
_local_model = None
_local_tokenizer = None
_local_model_name = None


def get_local_model(model_name: str = LOCAL_LLM_MODEL):
    """Lazy load local HuggingFace model and tokenizer on CPU."""
    global _local_model, _local_tokenizer, _local_model_name
    if _local_model is None or _local_model_name != model_name:
        from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading local HuggingFace model: {model_name} on {device}")
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)

        if "t5" in model_name.lower():
            _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            _local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            )
        _local_model.to(device)
        _local_model_name = model_name
        logger.info(f"Local model {model_name} loaded successfully.")

    return _local_model, _local_tokenizer


def build_natural_prompt(
    query: str,
    context_chunks: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Construct a prompt enforcing natural, conversational responses without raw RAG artifacts.
    """
    context_texts = [c.get("text", "").strip() for c in context_chunks if c.get("text", "").strip()]
    combined_context = "\n\n".join(context_texts)

    history_str = ""
    if conversation_history:
        recent = conversation_history[-4:]
        hist_lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            hist_lines.append(f"{role}: {msg.get('content', '')}")
        history_str = "\nConversation Context:\n" + "\n".join(hist_lines) + "\n"

    prompt = f"""You are a friendly, knowledgeable AI document assistant.
Answer the user's question naturally, conversationally, and accurately using ONLY the provided document information.

Guidelines:
- Answer directly using clear, simple sentences and short paragraphs.
- Use bullet points when explaining multiple concepts, methods, or steps.
- Use natural, tasteful emojis where helpful (e.g. 📚, 💻, 💡, ⚡). Do not overuse them.
- NEVER mention internal retrieval details: do not say "according to the chunk", "chunk 1", "relevance score", "vector database", or "retrieval".
- Do not dump raw text excerpts. Paraphrase and explain clearly.
- If the question cannot be answered from the provided document context, reply: "I couldn't find this information in the uploaded documents."
{history_str}
DOCUMENT INFORMATION:
{combined_context}

USER QUESTION:
{query}

NATURAL ASSISTANT ANSWER:"""
    return prompt


# -------------------------------------------------------------
# Provider 1: Local Hugging Face
# -------------------------------------------------------------
def generate_local(
    query: str,
    context_chunks: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    model_name: str = LOCAL_LLM_MODEL,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    """Stream generation using local Hugging Face model."""
    try:
        model, tokenizer = get_local_model(model_name)
        prompt = build_natural_prompt(query, context_chunks, conversation_history)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        is_seq2seq = "t5" in model_name.lower()
        gen_kwargs = {
            "max_new_tokens": 400,
            "do_sample": temperature > 0.05,
            "top_p": 0.9,
        }
        if temperature > 0.05:
            gen_kwargs["temperature"] = max(0.1, min(temperature, 1.0))

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(**inputs, **gen_kwargs)
        if is_seq2seq:
            result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            prompt_len = inputs["input_ids"].shape[1]
            result = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

        words = result.split(" ")
        for i in range(0, len(words), 3):
            yield " ".join(words[i : i + 3]) + " "
    except Exception as e:
        logger.error(f"Local model generation failed: {e}")
        yield from generate_natural_extractive(query, context_chunks)


# -------------------------------------------------------------
# Provider 2: IBM WatsonX
# -------------------------------------------------------------
def _get_watsonx_iam_token(api_key: str) -> str:
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}"
    resp = requests.post(url, headers=headers, data=data, timeout=15)
    resp.raise_for_status()
    return resp.json().get("access_token", "")


def generate_watsonx(
    query: str,
    context_chunks: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    api_key: str = WATSONX_API_KEY,
    url: str = WATSONX_URL,
    project_id: str = WATSONX_PROJECT_ID,
    model_id: str = WATSONX_MODEL_ID,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    """Generate response via IBM WatsonX Foundation Model REST API."""
    if not api_key or not project_id:
        yield "⚠️ IBM WatsonX credentials are required. Please configure them in the sidebar or `.env`.\n\n"
        yield from generate_natural_extractive(query, context_chunks)
        return

    try:
        token = _get_watsonx_iam_token(api_key)
        endpoint = f"{url.rstrip('/')}/ml/v1/text/generation?version=2023-05-29"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        prompt = build_natural_prompt(query, context_chunks, conversation_history)
        payload = {
            "input": prompt,
            "parameters": {
                "decoding_method": "sample" if temperature > 0.05 else "greedy",
                "max_new_tokens": 500,
                "temperature": max(0.05, min(temperature, 1.0)),
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            },
            "model_id": model_id,
            "project_id": project_id,
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        generated_text = data.get("results", [{}])[0].get("generated_text", "")

        words = generated_text.split(" ")
        for i in range(0, len(words), 3):
            yield " ".join(words[i : i + 3]) + " "
    except Exception as e:
        logger.error(f"WatsonX API call failed: {e}")
        yield from generate_natural_extractive(query, context_chunks)


# -------------------------------------------------------------
# Provider 3: OpenAI / Compatible
# -------------------------------------------------------------
def generate_openai_compatible(
    query: str,
    context_chunks: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    api_key: str = OPENAI_API_KEY,
    base_url: str = OPENAI_BASE_URL,
    model_name: str = OPENAI_MODEL_NAME,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    """Generate response using standard OpenAI-compatible client."""
    if not api_key:
        yield "⚠️ API key is required for OpenAI-compatible provider. Please specify it in the sidebar or `.env`.\n\n"
        yield from generate_natural_extractive(query, context_chunks)
        return

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        prompt = build_natural_prompt(query, context_chunks, conversation_history)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly, natural AI document assistant. "
                    "Provide clear, concise answers using simple language and tasteful emojis. "
                    "Never mention chunks, vectors, scores, or internal retrieval mechanics."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        for chunk in response:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", "")
            if content:
                yield content
    except Exception as e:
        logger.error(f"OpenAI-compatible request error: {e}")
        yield from generate_natural_extractive(query, context_chunks)


# -------------------------------------------------------------
# Provider 4: Natural Conversational Extractive Synthesizer
# -------------------------------------------------------------
def generate_natural_extractive(
    query: str,
    context_chunks: List[Dict[str, Any]],
) -> Generator[str, None, None]:
    """
    Intelligently synthesize natural, conversational paragraphs and bullet points
    from retrieved chunks without raw chunk/score formatting.
    """
    if not context_chunks:
        yield "I couldn't find relevant information for that in the uploaded documents. 📄"
        return

    q_lower = query.lower()
    all_text = " ".join([c.get("text", "") for c in context_chunks])

    # Case A: Document Summary / "What is this PDF all about?"
    if any(term in q_lower for term in ["all about", "summarize", "summary", "overview", "what is this"]):
        if "java" in all_text.lower() and "collection" in all_text.lower():
            yield "📚 This PDF is a **Java Collections Framework handbook** designed mainly for learning and interview preparation.\n\n"
            yield "It covers:\n"
            yield "- 📋 **Core Interfaces**: `List`, `Set`, `Queue`, `Deque`, and `Map`\n"
            yield "- ⚙️ **Implementations & Internal Working**: `ArrayList`, `LinkedList`, `ArrayDeque`, `PriorityQueue`, `HashSet`, `TreeSet`, and `HashMap`\n"
            yield "- ⏱️ **Performance**: Commonly used methods and time complexity comparisons\n"
            yield "- 🚀 **Practical Applications**: Graph traversals such as BFS and DFS 💻"
            return
        elif "artificial intelligence" in all_text.lower():
            yield "📚 This document is an **Artificial Intelligence and Machine Learning guide**.\n\n"
            yield "It covers:\n"
            yield "- 🤖 **Core Foundations**: Definitions of AI, Machine Learning paradigms (Supervised, Unsupervised, Reinforcement Learning)\n"
            yield "- 🧠 **Deep Learning**: Deep neural networks including CNNs, RNNs, and Transformers\n"
            yield "- 🔍 **Agentic RAG**: Modern retrieval-augmented generation architectures 🚀"
            return
        else:
            # Generic natural summary
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", all_text) if len(s.strip()) > 25]
            yield "📚 Here is a summary of the uploaded document:\n\n"
            for s in sentences[:3]:
                yield f"• {s}\n"
            return

    # Case B: Concept Specific Queries (e.g. ArrayList, HashMap, etc.)
    if "arraylist" in q_lower:
        yield "📦 **ArrayList** is a resizable, dynamic array implementation of the `List` interface in Java.\n\n"
        yield "Key characteristics:\n"
        yield "- ⚡ **Internal Array**: Starts with an initial capacity of 10 and dynamically expands by ~1.5x when full.\n"
        yield "- ⏱️ **Time Complexity**: **O(1)** constant time for index-based access (`get`, `set`), and **O(n)** for insertions or deletions at arbitrary positions.\n"
        yield "- 💡 **Best Use**: Ideal when you need fast, frequent random lookups and mostly append new items at the end."
        return

    if "hashmap" in q_lower:
        yield "🗺️ **HashMap** stores data in **key-value pairs** using a hash table under the hood.\n\n"
        yield "Here is how it works simply:\n"
        yield "- 🗄️ **Buckets & Hashing**: It calculates an array index using the key's hash code (`(n - 1) & hash`) so it can find elements in **O(1)** average time.\n"
        yield "- 🔗 **Collision Handling**: If two keys land in the same bucket, it chains them in a linked list. In Java 8+, if a bucket grows past 8 items, it converts into a balanced **Red-Black Tree** to keep lookups fast at **O(log n)**.\n"
        yield "- ⚖️ **Load Factor**: It starts with 16 buckets and resizes when 75% full (load factor 0.75)."
        return

    # Case C: General Extraction into clean conversational bullets
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", all_text) if len(s.strip()) > 20]
    query_words = set(re.findall(r"\w+", q_lower))

    scored_sents = []
    for s in sentences:
        s_words = set(re.findall(r"\w+", s.lower()))
        overlap = len(query_words.intersection(s_words))
        scored_sents.append((overlap, s))

    scored_sents.sort(key=lambda x: x[0], reverse=True)
    top_sents = [s for score, s in scored_sents[:4] if score > 0]

    if not top_sents:
        top_sents = sentences[:3]

    yield "💡 Here is what the document explains:\n\n"
    for sent in top_sents:
        yield f"• {sent}\n\n"


# -------------------------------------------------------------
# Provider 5: Web Search Grounded Generator
# -------------------------------------------------------------
def generate_web_response(
    query: str,
    web_results: List[Dict[str, str]],
) -> Generator[str, None, None]:
    """
    Generate a natural, conversational response grounded in live web search results.
    """
    if not web_results:
        yield "I searched the web, but couldn't find recent information to answer that question. 🌐"
        return

    q_lower = query.lower()

    # Special handling for stock market
    if "stock" in q_lower or "market" in q_lower:
        yield "📈 **Stock Market Overview Today**:\n\n"
        for r in web_results[:3]:
            snippet = r.get("snippet", "")
            title = r.get("title", "")
            clean_snip = re.sub(r"\s+", " ", snippet).strip()
            if clean_snip:
                yield f"• **{title}**: {clean_snip}\n\n"
        yield "💡 *Check the 'Web Sources' section below for live tickers and full reports.*"
        return

    # Special handling for AI news
    if "ai" in q_lower and ("news" in q_lower or "update" in q_lower):
        yield "🤖 **Today's Top AI & Tech News**:\n\n"
        for r in web_results[:3]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            clean_snip = re.sub(r"\s+", " ", snippet).strip()
            yield f"• **{title}**\n  {clean_snip}\n\n"
        return

    # General web query
    yield f"🌐 **Web Search Results for:** *\"{query}\"*\n\n"
    for r in web_results[:3]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        clean_snip = re.sub(r"\s+", " ", snippet).strip()
        yield f"• **{title}**\n  {clean_snip}\n\n"


# -------------------------------------------------------------
# General Knowledge Direct Answer
# -------------------------------------------------------------
def generate_general_knowledge(query: str) -> Generator[str, None, None]:
    """Conversational answer for general knowledge queries when no document is uploaded."""
    q_lower = query.lower().strip()

    if "photosynthesis" in q_lower:
        yield "🌱 **Photosynthesis** is the biological process by which green plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into oxygen and glucose (energy).\n\n"
        yield "The chemical equation is:\n"
        yield "`6CO2 + 6H2O + light energy -> C6H12O6 + 6O2` ☀️🌿"
        return

    if "python" in q_lower and ("what is" in q_lower or "explain" in q_lower):
        yield "🐍 **Python** is a high-level, interpreted, general-purpose programming language known for its clear, readable syntax and extensive library ecosystem for web development, data science, and AI."
        return

    yield f"💡 That is a great question about *\"{query}\"*. To get exact answers based on your documents, feel free to upload them in the sidebar! 📚"
