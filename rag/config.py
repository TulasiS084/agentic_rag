"""
Central Configuration for the Agentic RAG Application.
Loads environment variables and sets up project paths and hyperparameters.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load from .env if present
load_dotenv()

# Root and Data Directories
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DB_DIR = DATA_DIR / "chroma"
CACHE_DIR = DATA_DIR / "cache"

# Ensure all runtime directories exist
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Hyperparameters
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# Retrieval & Fusion
DEFAULT_DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "10"))
DEFAULT_SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "10"))
DEFAULT_TOP_N_RERANK = int(os.getenv("TOP_N_RERANK", "5"))
DEFAULT_LLM_TOP_K = int(os.getenv("LLM_TOP_K", "4"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Reranker Model
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# IBM WatsonX Configuration
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-chat-v2")

# OpenAI / Compatible Providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# Local Hugging Face Model
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-base")

# ChromaDB Prefix
CHROMA_COLLECTION_PREFIX = "doc"
