"""
NotebookLM-Style AI Document Assistant.
Full single-application Agentic RAG system built with Python and Streamlit.
Features natural conversational responses, smart document gating, and automatic web search fallback.
"""
from __future__ import annotations
import os
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

# Import RAG pipeline modules
from rag import (
    config,
    loaders,
    chunker,
    embeddings,
    vectorstore,
    retriever,
    reranker,
    generator,
    agent,
    voice,
    web_search,
)

# -----------------------------------------------------------------------------
# Custom CSS for Modern NotebookLM Aesthetics
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .app-header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563EB, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-header-sub {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .source-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.12);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .score-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Model Resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Embedding Model (all-MiniLM-L6-v2)...")
def load_embedding_engine():
    return embeddings.get_embedding_model()


@st.cache_resource(show_spinner="Loading Cross-Encoder Reranker...")
def load_reranker_engine():
    return reranker.get_reranker_model()


load_embedding_engine()
load_reranker_engine()


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "documents" not in st.session_state:
    st.session_state.documents = {}  # doc_id -> metadata dict

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = None


# -----------------------------------------------------------------------------
# Document Processing Helper
# -----------------------------------------------------------------------------
def process_uploaded_file(uploaded_file, chunk_size: int, chunk_overlap: int) -> Optional[str]:
    """Save, extract, chunk, embed, and index an uploaded document."""
    filename = uploaded_file.name
    save_path = config.UPLOAD_DIR / filename

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_size = save_path.stat().st_size
    file_hash = vectorstore.compute_file_hash(save_path)

    for doc_id, doc_meta in st.session_state.documents.items():
        if doc_meta.get("file_hash") == file_hash and doc_meta.get("status") == "READY":
            st.info(f"ℹ️ Document '{filename}' is already indexed.")
            return doc_id

    doc_id = str(uuid.uuid4())[:8]
    suffix = save_path.suffix.lower()

    progress_box = st.empty()
    progress_bar = st.progress(0)
    start_time = time.time()

    try:
        progress_box.markdown(f"**Step 1/5**: 📖 Extracting text from `{filename}`...")
        progress_bar.progress(20)
        pages = loaders.load_document(save_path)

        if not pages:
            progress_box.error(f"⚠️ No text could be extracted from `{filename}`.")
            return None

        progress_box.markdown(f"**Step 2/5**: 🧩 Splitting into semantic chunks (size={chunk_size}, overlap={chunk_overlap})...")
        progress_bar.progress(40)
        chunks, stats = chunker.chunk_pages(
            pages=pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_id=doc_id,
        )

        if not chunks:
            progress_box.error(f"⚠️ Chunking produced 0 chunks for `{filename}`.")
            return None

        progress_box.markdown(f"**Step 3/5**: 🧠 Generating MiniLM vector embeddings ({len(chunks)} chunks)...")
        progress_bar.progress(60)
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = embeddings.embed_texts(chunk_texts)

        progress_box.markdown(f"**Step 4/5**: 💾 Indexing vectors in persistent ChromaDB...")
        progress_bar.progress(80)
        chunk_ids = [c["chunk_id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        vectorstore.add_chunks(
            document_id=doc_id,
            chunk_ids=chunk_ids,
            texts=chunk_texts,
            embeddings=chunk_embeddings,
            metadatas=metadatas,
        )

        progress_box.markdown(f"**Step 5/5**: 🔍 Building BM25 sparse lexical index...")
        progress_bar.progress(95)
        retriever.build_bm25_index(document_id=doc_id, chunks=chunks)

        elapsed = round(time.time() - start_time, 2)
        progress_bar.progress(100)
        progress_box.success(f"✅ Ingestion complete for **{filename}** in {elapsed}s! ({len(chunks)} chunks, {len(pages)} pages/records)")
        time.sleep(1.0)
        progress_box.empty()
        progress_bar.empty()

        st.session_state.documents[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "file_type": suffix.lstrip("."),
            "file_size": file_size,
            "file_hash": file_hash,
            "num_pages": len(pages),
            "num_chunks": len(chunks),
            "processing_time": elapsed,
            "status": "READY",
            "stats": stats,
        }
        return doc_id

    except Exception as e:
        progress_bar.empty()
        progress_box.error(f"❌ Failed to process `{filename}`: {str(e)}")
        logger.error(f"Ingestion error for {filename}: {e}", exc_info=True)
        return None


def delete_document_record(doc_id: str):
    """Delete a document from ChromaDB, BM25, and session state."""
    doc = st.session_state.documents.get(doc_id)
    if not doc:
        return

    vectorstore.delete_collection(doc_id)
    retriever.remove_bm25_index(doc_id)
    file_path = config.UPLOAD_DIR / doc["filename"]
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

    del st.session_state.documents[doc_id]
    st.toast(f"Removed document: {doc['filename']}")


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Document Workspace")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md", "csv", "pptx", "xlsx"],
        accept_multiple_files=True,
        help="Upload PDF (up to ~1000 pages), DOCX, TXT, MD, CSV, PPTX, or XLSX documents.",
    )

    with st.expander("⚙️ Ingestion & Chunking Settings", expanded=False):
        chunk_size = st.slider("Chunk Size", min_value=300, max_value=2500, value=config.DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=config.DEFAULT_CHUNK_OVERLAP, step=25)

    if uploaded_files:
        for uf in uploaded_files:
            already_indexed = any(d.get("filename") == uf.name for d in st.session_state.documents.values())
            if not already_indexed:
                process_uploaded_file(uf, chunk_size, chunk_overlap)

    active_docs = list(st.session_state.documents.values())
    st.markdown(f"#### Active Documents ({len(active_docs)})")

    if active_docs:
        for doc in active_docs:
            col_info, col_del = st.columns([0.82, 0.18])
            with col_info:
                st.markdown(
                    f"**✓ {doc['filename']}**  \n"
                    f"<small style='color: #94A3B8;'>{doc['file_type'].upper()} • {doc['num_pages']} pgs • {doc['num_chunks']} chunks</small>",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑️", key=f"del_{doc['id']}", help=f"Delete {doc['filename']}"):
                    delete_document_record(doc['id'])
                    st.rerun()
            st.divider()
    else:
        st.caption("No documents loaded yet. Chatbot works with Web Search & General Knowledge!")

    with st.expander("🔍 Retrieval & Reranker Settings", expanded=False):
        dense_top_k = st.slider("Dense Top-K", 2, 25, config.DEFAULT_DENSE_TOP_K)
        sparse_top_k = st.slider("Sparse Top-K", 2, 25, config.DEFAULT_SPARSE_TOP_K)
        top_n_rerank = st.slider("Reranker Top-N", 1, 15, config.DEFAULT_TOP_N_RERANK)
        llm_top_k = st.slider("Context Chunks to LLM", 1, 8, config.DEFAULT_LLM_TOP_K)
        use_cross_encoder = st.toggle("Use Cross-Encoder Reranker", value=True)

    with st.expander("🤖 Assistant Response Settings", expanded=False):
        llm_provider = st.selectbox(
            "Answer Engine",
            options=["Extractive Grounding", "Local HuggingFace", "IBM WatsonX", "OpenAI / Compatible"],
            index=0,
            help="Extractive Grounding generates fast, structured conversational answers directly without external API keys.",
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

        watsonx_config = None
        openai_config = None
        local_model_name = "google/flan-t5-base"

        if llm_provider == "Local HuggingFace":
            local_model_name = st.selectbox(
                "Local Model",
                options=["google/flan-t5-base", "google/flan-t5-small", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
                index=0,
            )
        elif llm_provider == "IBM WatsonX":
            w_key = st.text_input("WatsonX API Key", value=config.WATSONX_API_KEY, type="password")
            w_proj = st.text_input("WatsonX Project ID", value=config.WATSONX_PROJECT_ID)
            w_model = st.text_input("WatsonX Model ID", value=config.WATSONX_MODEL_ID)
            w_url = st.text_input("WatsonX URL", value=config.WATSONX_URL)
            watsonx_config = {"api_key": w_key, "project_id": w_proj, "model_id": w_model, "url": w_url}
        elif llm_provider == "OpenAI / Compatible":
            o_key = st.text_input("API Key", value=config.OPENAI_API_KEY, type="password")
            o_base = st.text_input("Base URL", value=config.OPENAI_BASE_URL)
            o_model = st.text_input("Model Name", value=config.OPENAI_MODEL_NAME)
            openai_config = {"api_key": o_key, "base_url": o_base, "model_name": o_model}

    st.markdown("### 🎙️ Voice Assistant")
    enable_voice_answers = st.checkbox("Generate Voice Answers (TTS)", value=False)

    st.markdown("### 🛠️ Actions")
    col_reb, col_clr = st.columns(2)
    with col_reb:
        if st.button("🔄 Rebuild", use_container_width=True):
            for doc in list(st.session_state.documents.values()):
                chunks = vectorstore.get_all_chunks(doc["id"])
                if chunks:
                    retriever.build_bm25_index(doc["id"], chunks)
            st.success("Rebuilt index!")
    with col_clr:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# -------------------------------------------------------------
# MAIN CONTENT AREA
# -------------------------------------------------------------
st.markdown('<div class="app-header-title">AI Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="app-header-sub">Upload documents, explore their contents, and ask questions using AI.</div>', unsafe_allow_html=True)

# Dashboard Metrics
total_docs = len(st.session_state.documents)
total_pages = sum(d.get("num_pages", 0) for d in st.session_state.documents.values())
total_chunks = sum(d.get("num_chunks", 0) for d in st.session_state.documents.values())

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Documents</div><div class="metric-value">{total_docs}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Pages / Recs</div><div class="metric-value">{total_pages}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Chunks</div><div class="metric-value">{total_chunks}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-label">Embedding</div><div class="metric-value" style="font-size: 1.05rem;">MiniLM-L6</div></div>', unsafe_allow_html=True)
with m5:
    st.markdown('<div class="metric-card"><div class="metric-label">Vector Store</div><div class="metric-value" style="font-size: 1.05rem;">ChromaDB</div></div>', unsafe_allow_html=True)
with m6:
    st.markdown('<div class="metric-card"><div class="metric-label">Fallback</div><div class="metric-value" style="font-size: 1.05rem;">Web Search 🌐</div></div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------------
# Chat Area & History Rendering
# -------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 🔎 Optional Expandable Document Sources
        sources = msg.get("sources", [])
        if sources:
            with st.expander("🔎 View Sources", expanded=False):
                for s in sources:
                    st.markdown(f"**📄 Document:** {s.get('source', 'Unknown')} | **Page:** {s.get('page', 'N/A')} | **Score:** `{s.get('score', 0):.4f}`")
                    st.markdown(f"> {s.get('text', '')}")
                    st.divider()

        # 🌐 Optional Expandable Web Sources
        web_sources = msg.get("web_sources", [])
        if web_sources:
            with st.expander("🌐 Web Sources", expanded=False):
                for ws in web_sources:
                    title = ws.get("title", "Web Source")
                    url = ws.get("url", "#")
                    st.markdown(f"• [🔗 **{title}**]({url})")
                    snippet = ws.get("snippet", "")
                    if snippet:
                        st.caption(snippet)

        if msg.get("audio_bytes"):
            st.audio(msg["audio_bytes"], format="audio/mp3")


# -------------------------------------------------------------
# Voice Input Widget
# -------------------------------------------------------------
voice_col1, voice_col2 = st.columns([0.85, 0.15])
with voice_col2:
    st.markdown("**🎙️ Speak Question**")
    recorded_audio = st.audio_input("Record your question", key="audio_recorder", label_visibility="collapsed")

voice_query_text = None
if recorded_audio is not None:
    audio_data = recorded_audio.read()
    audio_hash = hash(audio_data)
    if st.session_state.last_transcription != audio_hash:
        st.session_state.last_transcription = audio_hash
        with st.spinner("🎙️ Transcribing voice..."):
            text, err = voice.transcribe_audio(audio_data)
            if text:
                st.success(f"Transcribed: *\"{text}\"*")
                voice_query_text = text
            elif err:
                st.warning(f"Voice input: {err}")


# -------------------------------------------------------------
# Chat Input & Routing
# -------------------------------------------------------------
text_query = st.chat_input("Ask a question, e.g. 'What is this PDF all about?', 'What is ArrayList?', 'What is the stock market today?'...")

user_query = voice_query_text if voice_query_text else text_query

if user_query:
    active_doc_ids = list(st.session_state.documents.keys())

    # User message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant response
    with st.chat_message("assistant"):
        provider_key = "extractive"
        if llm_provider == "IBM WatsonX":
            provider_key = "watsonx"
        elif llm_provider == "OpenAI / Compatible":
            provider_key = "openai"
        elif llm_provider == "Local HuggingFace":
            provider_key = "local"

        with st.spinner("Thinking..."):
            pipeline_result = agent.run_agentic_rag(
                query=user_query,
                document_ids=active_doc_ids,
                conversation_history=st.session_state.messages[:-1],
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                top_n_rerank=top_n_rerank,
                llm_top_k=llm_top_k,
                provider=provider_key,
                model_name=local_model_name,
                temperature=temperature,
                use_cross_encoder=use_cross_encoder,
                watsonx_config=watsonx_config,
                openai_config=openai_config,
            )

        # Stream natural answer without raw RAG info
        stream_gen = pipeline_result["stream_generator"]
        full_response = st.write_stream(stream_gen)

        # 🔎 Optional Expandable Sources
        retrieved_sources = pipeline_result.get("retrieved_sources", [])
        if retrieved_sources:
            with st.expander("🔎 View Sources", expanded=False):
                for s in retrieved_sources:
                    st.markdown(f"**📄 Document:** {s.get('source', 'Unknown')} | **Page:** {s.get('page', 'N/A')} | **Score:** `{s.get('score', 0):.4f}`")
                    st.markdown(f"> {s.get('text', '')}")
                    st.divider()

        # 🌐 Optional Expandable Web Sources
        web_sources = pipeline_result.get("web_sources", [])
        if web_sources:
            with st.expander("🌐 Web Sources", expanded=False):
                for ws in web_sources:
                    title = ws.get("title", "Web Source")
                    url = ws.get("url", "#")
                    st.markdown(f"• [🔗 **{title}**]({url})")
                    snippet = ws.get("snippet", "")
                    if snippet:
                        st.caption(snippet)

        # Voice output if enabled
        tts_bytes = None
        if enable_voice_answers and full_response:
            with st.spinner("🔊 Generating voice audio..."):
                tts_bytes = voice.text_to_speech_bytes(full_response)
                if tts_bytes:
                    st.audio(tts_bytes, format="audio/mp3")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": retrieved_sources,
            "web_sources": web_sources,
            "stats": pipeline_result.get("stats", {}),
            "audio_bytes": tts_bytes,
        })
