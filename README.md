# AI Document Assistant — Production-Quality Agentic RAG

A modern, high-performance **NotebookLM-style document-based AI assistant** built entirely with **Python + Streamlit**. The application features an end-to-end Agentic RAG pipeline with multi-format ingestion, persistent ChromaDB indexing, hybrid dense-sparse retrieval (BM25 + ChromaDB via Reciprocal Rank Fusion), Cross-Encoder reranking, modular LLM generation (Local HuggingFace, IBM WatsonX, OpenAI-compatible, and extractive fallback), conversational memory, and voice interaction.

---

## Key Features

- **NotebookLM-Style Experience**: Unified Streamlit web application with document management, interactive citation badges, ingestion stats, and real-time streaming answers.
- **Natural Conversational Responses**:
  - Responses behave like a modern, friendly AI assistant using simple sentences, short paragraphs, bullet points, and appropriate emojis (e.g. 📚, 💻, 🚀).
  - Internal RAG artifacts (chunk IDs, rank numbers, raw similarity/relevance scores, raw chunk text) are **completely hidden** from the main chatbot answer.
- **Dedicated Expandable Sources**:
  - **Document Queries**: Optional `🔎 View Sources` drawer displaying document name, page number, retrieved text, and relevance score.
  - **Web Queries**: Optional `🌐 Web Sources` drawer displaying clickable links and snippets.
- **Smart Relevance Gating & Automatic Web Search Fallback**:
  - Gated using Cross-Encoder relevance score thresholds to verify if questions belong to the document.
  - Automatically triggers web search for current/time-sensitive queries (e.g. stock market today, latest news) and out-of-document questions without manual toggles.
- **Multi-Format Streaming Ingestion**:
  - Supports **PDF** (streaming up to ~1000 pages with PyMuPDF/pypdf), **DOCX**, **TXT**, **Markdown**, **CSV**, **PPTX**, and **XLSX**.
  - Automatic text normalization, repair of hyphenated line breaks, and null-byte sanitization while preserving paragraph semantics.
- **Persistent Vector Store & Duplicate Detection**:
  - Stores vector embeddings in persistent ChromaDB collections.
  - Document SHA-256 hashing skips redundant re-embedding.
  - Individual document deletion and instant index rebuilding.
- **Hybrid Retrieval (Dense + Sparse)**:
  - **Dense Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` embeddings with cosine distance.
  - **Sparse Retrieval**: `BM25Okapi` with disk-cached token indices.
  - **Reciprocal Rank Fusion (RRF)**: Combines dense and sparse ranked lists using $RRF(d) = \sum \frac{1}{60 + rank(d)}$.
- **Cross-Encoder Reranking**:
  - Scores candidate chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - Includes a lightweight fallback ranking mode for low-resource CPUs.
- **Agentic Gating & Context Selection**:
  - Intent classification: detects greetings, summaries, chapter-specific queries, comparisons, and follow-ups.
  - Resolves pronouns and references in follow-up queries using conversation history.
  - Passes only the top $K$ verified chunks with clear source citations (`[Source: filename, Page: X]`).
- **Modular LLM Generator**:
  - **Local Hugging Face**: `google/flan-t5-base`, `google/flan-t5-small`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (runs locally without API keys).
  - **IBM WatsonX**: REST API integration for IBM Granite models (`ibm/granite-13b-chat-v2`).
  - **OpenAI-Compatible**: Seamless connection to OpenAI, Groq, Ollama, OpenRouter.
  - **Extractive Grounding**: Zero-dependency structured synthesis with direct excerpts and page citations.
- **Voice Assistant**:
  - **Speech-to-Text (STT)**: Voice input via microphone (`st.audio_input` + `SpeechRecognition`).
  - **Text-to-Speech (TTS)**: Spoken audio answers using `gTTS` with in-browser audio playback.

---

## Project Structure

```text
agentic_rag_2/
├── app.py                      # Streamlit application (NotebookLM-style UI)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── README.md                   # Complete documentation
│
├── rag/
│   ├── __init__.py
│   ├── config.py               # Settings, default hyperparameters & paths
│   ├── loaders.py              # Streaming multi-format document parser
│   ├── chunker.py              # RecursiveCharacterTextSplitter + dataset statistics
│   ├── embeddings.py           # all-MiniLM-L6-v2 singleton & batch encoding
│   ├── vectorstore.py          # Persistent ChromaDB collection management
│   ├── retriever.py            # Hybrid retrieval (ChromaDB + BM25 + RRF)
│   ├── reranker.py             # Cross-Encoder (ms-marco-MiniLM-L-6-v2) + fast fallback
│   ├── generator.py            # Modular LLM (Local HF, WatsonX, OpenAI, Extractive)
│   ├── agent.py                # Agentic query analyzer & contextual router
│   └── voice.py                # Speech recognition & text-to-speech audio bridge
│
├── data/
│   ├── uploads/                # Preserved user uploaded documents
│   ├── chroma/                 # ChromaDB persistent directory
│   └── cache/                  # Disk-cached BM25 indices
│
├── sample_docs/                # Sample test documents (PDF, DOCX, CSV, TXT)
├── scripts/
│   └── generate_samples.py     # Generator script for test documents
└── tests/
    ├── test_rag.py             # Pipeline unit & integration tests
    └── test_multi_format.py    # Multi-document end-to-end verification
```

---

## Installation & Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration (Optional)
Copy `.env.example` to `.env` to configure API keys for IBM WatsonX or OpenAI:
```bash
cp .env.example .env
```
*(Note: You can also configure all keys and endpoints directly in the Streamlit sidebar at runtime, or use the built-in local Hugging Face and Extractive modes without any API keys!)*

### 4. Run the Application
Launch the unified Streamlit interface:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## How the Agentic RAG Pipeline Works

```text
User Query (Text or Voice)
           ↓
   Intent Classifier
   [Greeting | Summary | Chapter | Comparison | Follow-Up | Doc Query]
           ↓
   Retrieval Gating (Skip retrieval for chitchat/empty docs)
           ↓
   Query Rewriter (Resolves pronouns like "explain it" via chat history)
           ↓
   Hybrid Search
   ├── Dense Retrieval (ChromaDB + all-MiniLM-L6-v2 embeddings)
   └── Sparse Retrieval (BM25Okapi lexical matching)
           ↓
   Reciprocal Rank Fusion (RRF)
   Merges candidate chunks & removes duplicates
           ↓
   Cross-Encoder Reranker
   (ms-marco-MiniLM-L-6-v2 scores query-chunk pairs)
           ↓
   Context Assembly
   Selects top-K highest-ranked chunks with [Source, Page, ID] headers
           ↓
   LLM Generation (Local HF / WatsonX / OpenAI / Extractive)
           ↓
   Answer + Expandable Source Citations (+ Voice Audio)
```

---

## Testing & Verification

Run the automated test suites to verify that all components operate properly:

```bash
# Run core pipeline test
python tests/test_rag.py

# Run multi-format end-to-end test (PDF, DOCX, TXT, CSV)
python tests/test_multi_format.py
```
