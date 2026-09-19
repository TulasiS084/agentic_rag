"""
NotebookLM-Style AI Document Assistant with Synchronized PDF Viewer.
Features exact factual extraction, dual-column chat & visual document viewer,
automatic 10% chunk overlap, and real-time web search fallback.
"""
from __future__ import annotations
import os
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

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
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .app-header-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563EB, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .app-header-sub {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.0rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .viewer-card {
        background-color: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 12px;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Embedding Engine...")
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
    st.session_state.documents = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "viewer_doc_path" not in st.session_state:
    st.session_state.viewer_doc_path = None

if "viewer_doc_name" not in st.session_state:
    st.session_state.viewer_doc_name = ""

if "viewer_current_page" not in st.session_state:
    st.session_state.viewer_current_page = 1

if "viewer_total_pages" not in st.session_state:
    st.session_state.viewer_total_pages = 1

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = None


# -----------------------------------------------------------------------------
# Ingestion Helper (Large Document & Multi-Stage Optimized)
# -----------------------------------------------------------------------------
def process_uploaded_file(uploaded_file, chunk_size: int) -> Optional[str]:
    """Progressive page-by-page ingestion with live multi-stage feedback."""
    filename = uploaded_file.name
    save_path = config.UPLOAD_DIR / filename

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_size = save_path.stat().st_size
    file_hash = vectorstore.compute_file_hash(save_path)

    for doc_id, doc_meta in st.session_state.documents.items():
        if doc_meta.get("file_hash") == file_hash and doc_meta.get("status") == "READY":
            st.info(f"ℹ️ Document '{filename}' is already indexed.")
            # Set viewer target
            if save_path.suffix.lower() == ".pdf":
                st.session_state.viewer_doc_path = str(save_path)
                st.session_state.viewer_doc_name = filename
                st.session_state.viewer_total_pages = loaders.get_pdf_total_pages(save_path)
                st.session_state.viewer_current_page = 1
            return doc_id

    doc_id = str(uuid.uuid4())[:8]
    suffix = save_path.suffix.lower()

    progress_box = st.empty()
    progress_bar = st.progress(0)
    start_time = time.time()

    try:
        # Step 1: Page-by-page streaming extraction
        progress_box.markdown(f"**Step 1/4**: 📖 Streaming pages from `{filename}`...")
        progress_bar.progress(25)
        pages = loaders.load_document(save_path)

        if not pages:
            progress_box.error(f"⚠️ No text could be extracted from `{filename}`.")
            return None

        # Step 2: Chunking with automatic 10% overlap
        computed_overlap = chunker.compute_chunk_overlap(chunk_size)
        progress_box.markdown(
            f"**Step 2/4**: 🧩 Creating semantic chunks (size={chunk_size}, overlap={computed_overlap} [10%])  \n"
            f"*Pages processed: {len(pages)}*"
        )
        progress_bar.progress(50)
        chunks, stats = chunker.chunk_pages(
            pages=pages,
            chunk_size=chunk_size,
            document_id=doc_id,
            document_name=filename,
        )

        if not chunks:
            progress_box.error(f"⚠️ Chunking produced 0 chunks for `{filename}`.")
            return None

        # Step 3: Batch Embeddings
        progress_box.markdown(
            f"**Step 3/4**: 🧠 Generating MiniLM vector embeddings ({len(chunks)} chunks in batches of 64)...  \n"
            f"*Pages: {len(pages)} | Chunks created: {len(chunks)}*"
        )
        progress_bar.progress(75)
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = embeddings.embed_texts(chunk_texts)

        # Step 4: Persistent ChromaDB & BM25
        progress_box.markdown(f"**Step 4/4**: 💾 Updating persistent ChromaDB & BM25 indices...")
        progress_bar.progress(90)
        chunk_ids = [c["chunk_id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        vectorstore.add_chunks(
            document_id=doc_id,
            chunk_ids=chunk_ids,
            texts=chunk_texts,
            embeddings=chunk_embeddings,
            metadatas=metadatas,
        )
        retriever.build_bm25_index(document_id=doc_id, chunks=chunks)

        elapsed = round(time.time() - start_time, 2)
        progress_bar.progress(100)
        progress_box.success(
            f"✅ **{filename}** indexed in {elapsed}s! ({len(pages)} pages/records, {len(chunks)} chunks)"
        )
        time.sleep(1.0)
        progress_box.empty()
        progress_bar.empty()

        # Update session
        st.session_state.documents[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "path": str(save_path),
            "file_type": suffix.lstrip("."),
            "file_size": file_size,
            "file_hash": file_hash,
            "num_pages": len(pages),
            "num_chunks": len(chunks),
            "processing_time": elapsed,
            "status": "READY",
            "stats": stats,
        }

        # Initialize viewer if PDF
        if suffix == ".pdf":
            st.session_state.viewer_doc_path = str(save_path)
            st.session_state.viewer_doc_name = filename
            st.session_state.viewer_total_pages = len(pages)
            st.session_state.viewer_current_page = 1

        return doc_id

    except Exception as e:
        progress_bar.empty()
        progress_box.error(f"❌ Failed to process `{filename}`: {str(e)}")
        logger.error(f"Ingestion error for {filename}: {e}", exc_info=True)
        return None


def delete_document_record(doc_id: str):
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
    if st.session_state.viewer_doc_name == doc.get("filename"):
        st.session_state.viewer_doc_path = None
        st.session_state.viewer_doc_name = ""
    st.toast(f"Removed: {doc['filename']}")


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Document Workspace")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md", "csv", "pptx", "xlsx"],
        accept_multiple_files=True,
        help="Supports textbooks, papers, multi-page PDFs, DOCX, CSV, etc.",
    )

    with st.expander("⚙️ Ingestion & Chunking", expanded=False):
        chunk_size = st.slider("Chunk Size (chars)", min_value=400, max_value=2500, value=config.DEFAULT_CHUNK_SIZE, step=50)
        auto_overlap = chunker.compute_chunk_overlap(chunk_size)
        st.info(f"📐 **Overlap**: `{auto_overlap}` characters *(automatically 10%)*")

    if uploaded_files:
        for uf in uploaded_files:
            already_indexed = any(d.get("filename") == uf.name for d in st.session_state.documents.values())
            if not already_indexed:
                process_uploaded_file(uf, chunk_size)

    # Active Documents list
    active_docs = list(st.session_state.documents.values())
    st.markdown(f"#### Active Documents ({len(active_docs)})")

    if active_docs:
        for doc in active_docs:
            col_info, col_btn = st.columns([0.72, 0.28])
            with col_info:
                st.markdown(
                    f"**✓ {doc['filename']}**  \n"
                    f"<small style='color: #94A3B8;'>{doc['file_type'].upper()} • {doc['num_pages']} pgs • {doc['num_chunks']} chunks</small>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                # View in Document Viewer button
                if doc["file_type"] == "pdf":
                    if st.button("👁️", key=f"view_{doc['id']}", help="Open in Document Viewer"):
                        st.session_state.viewer_doc_path = doc.get("path")
                        st.session_state.viewer_doc_name = doc.get("filename")
                        st.session_state.viewer_total_pages = doc.get("num_pages", 1)
                        st.session_state.viewer_current_page = 1
                        st.rerun()
                if st.button("🗑️", key=f"del_{doc['id']}", help=f"Delete {doc['filename']}"):
                    delete_document_record(doc['id'])
                    st.rerun()
            st.divider()
    else:
        st.caption("No documents loaded yet. Chatbot is active with Web Search & General Knowledge.")

    with st.expander("🔍 Retrieval Candidate Pool", expanded=False):
        dense_top_k = st.slider("Dense Top-K (ChromaDB)", 5, 30, config.DEFAULT_DENSE_TOP_K)
        sparse_top_k = st.slider("Sparse Top-K (BM25)", 5, 30, config.DEFAULT_SPARSE_TOP_K)
        top_n_rerank = st.slider("Reranker Top-N", 2, 12, config.DEFAULT_TOP_N_RERANK)
        llm_top_k = st.slider("Context Chunks to LLM", 1, 8, config.DEFAULT_LLM_TOP_K)
        use_cross_encoder = st.toggle("Use Cross-Encoder Reranker", value=True)

    with st.expander("🤖 LLM Engine Settings", expanded=False):
        llm_provider = st.selectbox(
            "Answer Engine",
            options=["Extractive Grounding", "Local HuggingFace", "IBM WatsonX", "OpenAI / Compatible"],
            index=0,
            help="Extractive Grounding performs fast, exact factual extraction from retrieved context.",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

        watsonx_config = None
        openai_config = None
        local_model_name = "google/flan-t5-base"

        if llm_provider == "Local HuggingFace":
            local_model_name = st.selectbox("Local Model", ["google/flan-t5-base", "google/flan-t5-small", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
        elif llm_provider == "IBM WatsonX":
            w_key = st.text_input("API Key", config.WATSONX_API_KEY, type="password")
            w_proj = st.text_input("Project ID", config.WATSONX_PROJECT_ID)
            w_model = st.text_input("Model ID", config.WATSONX_MODEL_ID)
            w_url = st.text_input("URL", config.WATSONX_URL)
            watsonx_config = {"api_key": w_key, "project_id": w_proj, "model_id": w_model, "url": w_url}
        elif llm_provider == "OpenAI / Compatible":
            o_key = st.text_input("API Key", config.OPENAI_API_KEY, type="password")
            o_base = st.text_input("Base URL", config.OPENAI_BASE_URL)
            o_model = st.text_input("Model Name", config.OPENAI_MODEL_NAME)
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


# -----------------------------------------------------------------------------
# MAIN DASHBOARD HEADER & METRICS
# -----------------------------------------------------------------------------
st.markdown('<div class="app-header-title">AI Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="app-header-sub">Upload documents, explore their contents, and ask questions using AI.</div>', unsafe_allow_html=True)

# Metrics summary cards
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
    st.markdown('<div class="metric-card"><div class="metric-label">Vector DB</div><div class="metric-value" style="font-size: 1.05rem;">ChromaDB</div></div>', unsafe_allow_html=True)
with m6:
    st.markdown('<div class="metric-card"><div class="metric-label">Fallback</div><div class="metric-value" style="font-size: 1.05rem;">Web Search 🌐</div></div>', unsafe_allow_html=True)

st.divider()


# -----------------------------------------------------------------------------
# DUAL-COLUMN LAYOUT: CHAT (LEFT) & SYNCHRONIZED PDF VIEWER (RIGHT)
# -----------------------------------------------------------------------------
col_chat, col_viewer = st.columns([1.1, 0.9])

# =============================================================================
# COLUMN 1: CHAT & ASSISTANT INTERACTION
# =============================================================================
with col_chat:
    st.markdown("#### 💬 Conversation")

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 🔎 Optional Expandable Sources
            sources = msg.get("sources", [])
            if sources:
                with st.expander("🔎 View Sources", expanded=False):
                    for s in sources:
                        p_num = s.get("page", 1)
                        doc_src = s.get("source", "Document")
                        st.markdown(f"**📄 Document:** {doc_src} | **Page:** {p_num} | **Score:** `{s.get('score', 0):.4f}`")
                        st.markdown(f"> {s.get('text', '')}")
                        st.divider()

            # 🌐 Optional Expandable Web Sources
            web_sources = msg.get("web_sources", [])
            if web_sources:
                with st.expander("🌐 Web Sources", expanded=False):
                    for ws in web_sources:
                        st.markdown(f"• [🔗 **{ws.get('title', 'Web Source')}**]({ws.get('url', '#')})")
                        if ws.get("snippet"):
                            st.caption(ws.get("snippet"))

            if msg.get("audio_bytes"):
                st.audio(msg["audio_bytes"], format="audio/mp3")

    # Voice Input
    voice_col1, voice_col2 = st.columns([0.8, 0.2])
    with voice_col2:
        st.markdown("<small style='color:#94A3B8;'>🎙️ Voice</small>", unsafe_allow_html=True)
        recorded_audio = st.audio_input("Record", key="audio_recorder", label_visibility="collapsed")

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
                    st.warning(f"Voice: {err}")

    # Chat Input
    text_query = st.chat_input("Ask about your document, e.g. 'Name the authors', 'What is this PDF about?', 'What is the stock market today?'...")
    user_query = voice_query_text if voice_query_text else text_query

    if user_query:
        active_doc_ids = list(st.session_state.documents.keys())

        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant response
        with st.chat_message("assistant"):
            provider_key = "extractive"
            if llm_provider == "IBM WatsonX":
                provider_key = "watsonx"
            elif llm_provider == "OpenAI / Compatible":
                provider_key = "openai"
            elif llm_provider == "Local HuggingFace":
                provider_key = "local"

            with st.spinner("Searching document & reasoning..."):
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

            # Stream natural answer without raw RAG markers
            stream_gen = pipeline_result["stream_generator"]
            full_response = st.write_stream(stream_gen)

            # Synchronize Document Viewer with cited source page
            cited_page = pipeline_result.get("primary_page")
            cited_doc = pipeline_result.get("primary_doc")
            if cited_page:
                st.session_state.viewer_current_page = cited_page
                # Find matching PDF path if needed
                for d in st.session_state.documents.values():
                    if d.get("filename") == cited_doc and d.get("file_type") == "pdf":
                        st.session_state.viewer_doc_path = d.get("path")
                        st.session_state.viewer_doc_name = d.get("filename")
                        st.session_state.viewer_total_pages = d.get("num_pages", 1)
                        break

            # 🔎 Optional Expandable Sources
            retrieved_sources = pipeline_result.get("retrieved_sources", [])
            if retrieved_sources:
                with st.expander("🔎 View Sources", expanded=False):
                    for s in retrieved_sources:
                        p_num = s.get("page", 1)
                        st.markdown(f"**📄 Document:** {s.get('source', 'Unknown')} | **Page:** {p_num} | **Score:** `{s.get('score', 0):.4f}`")
                        st.markdown(f"> {s.get('text', '')}")
                        st.divider()

            # 🌐 Optional Expandable Web Sources
            web_sources = pipeline_result.get("web_sources", [])
            if web_sources:
                with st.expander("🌐 Web Sources", expanded=False):
                    for ws in web_sources:
                        st.markdown(f"• [🔗 **{ws.get('title', 'Web Source')}**]({ws.get('url', '#')})")
                        if ws.get("snippet"):
                            st.caption(ws.get("snippet"))

            tts_bytes = None
            if enable_voice_answers and full_response:
                with st.spinner("🔊 Generating audio..."):
                    tts_bytes = voice.text_to_speech_bytes(full_response)
                    if tts_bytes:
                        st.audio(tts_bytes, format="audio/mp3")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_sources,
                "web_sources": web_sources,
                "primary_page": cited_page,
                "stats": pipeline_result.get("stats", {}),
                "audio_bytes": tts_bytes,
            })
            st.rerun()


# =============================================================================
# COLUMN 2: SYNCHRONIZED DOCUMENT VIEWER
# =============================================================================
with col_viewer:
    st.markdown("#### 📄 Document Viewer")

    # If no viewer document is active, pick the first PDF in active documents
    if not st.session_state.viewer_doc_path:
        for d in st.session_state.documents.values():
            if d.get("file_type") == "pdf":
                st.session_state.viewer_doc_path = d.get("path")
                st.session_state.viewer_doc_name = d.get("filename")
                st.session_state.viewer_total_pages = d.get("num_pages", 1)
                st.session_state.viewer_current_page = 1
                break

    viewer_path = st.session_state.viewer_doc_path

    if viewer_path and Path(viewer_path).exists():
        total_pgs = st.session_state.viewer_total_pages or loaders.get_pdf_total_pages(viewer_path)
        curr_pg = max(1, min(st.session_state.viewer_current_page, total_pgs))

        # Viewer Header & Navigation Row
        st.markdown(f"**Viewing:** `{st.session_state.viewer_doc_name}`")

        nav_prev, nav_info, nav_next, nav_jump = st.columns([0.22, 0.35, 0.22, 0.21])
        with nav_prev:
            if st.button("◀ Prev", use_container_width=True, disabled=(curr_pg <= 1)):
                st.session_state.viewer_current_page = max(1, curr_pg - 1)
                st.rerun()
        with nav_info:
            st.markdown(f"<div style='text-align:center; padding-top:6px; font-weight:600;'>Page {curr_pg} of {total_pgs}</div>", unsafe_allow_html=True)
        with nav_next:
            if st.button("Next ▶", use_container_width=True, disabled=(curr_pg >= total_pgs)):
                st.session_state.viewer_current_page = min(total_pgs, curr_pg + 1)
                st.rerun()
        with nav_jump:
            jump_page = st.number_input("Jump", min_value=1, max_value=total_pgs, value=curr_pg, label_visibility="collapsed")
            if jump_page != curr_pg:
                st.session_state.viewer_current_page = int(jump_page)
                st.rerun()

        # Quick Jump buttons for recent sources
        recent_pages = set()
        for m in reversed(st.session_state.messages):
            if m.get("role") == "assistant":
                for s in m.get("sources", []):
                    p = s.get("page")
                    if isinstance(p, int):
                        recent_pages.add(p)
            if len(recent_pages) >= 4:
                break

        if recent_pages:
            st.markdown("<small style='color:#94A3B8;'>Jump to cited source page:</small>", unsafe_allow_html=True)
            jump_cols = st.columns(len(recent_pages))
            for i, p in enumerate(sorted(recent_pages)):
                with jump_cols[i]:
                    if st.button(f"📑 Page {p}", key=f"src_jump_{p}", use_container_width=True):
                        st.session_state.viewer_current_page = p
                        st.rerun()

        # Render PDF page to high-resolution image
        with st.spinner("Rendering page..."):
            page_png_bytes = loaders.render_pdf_page_to_png(viewer_path, curr_pg, dpi=130)

        if page_png_bytes:
            st.image(page_png_bytes, caption=f"Document Page {curr_pg}", use_container_width=True)
        else:
            st.warning("Could not render page image.")

    else:
        # Placeholder when no PDF is active
        st.markdown(
            """
            <div class="viewer-card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 2.5rem; margin-bottom: 10px;">📄</div>
                <div style="font-weight: 600; font-size: 1.1rem; color: #E2E8F0;">Document Viewer Ready</div>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 6px;">
                    Upload a PDF or select an active document in the sidebar.<br>
                    The relevant source page cited in answers will automatically display here for visual inspection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
