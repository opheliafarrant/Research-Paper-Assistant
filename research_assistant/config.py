"""
Single source of truth for every setting in the project
All other files import from here change a value once it updates everywhere 
"""

import os
from pathlib import Path

# Path
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data"
CACHE_DIR      = DATA_DIR / "cache"
PDF_DIR        = DATA_DIR / "pdfs"
INGESTED_FILE  = DATA_DIR / "ingested.txt"      # one paper_id per line

# Environment configuration for offline and local model stores
HF_CACHE_DIR   = BASE_DIR / ".hf-cache"
if HF_CACHE_DIR.exists():
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODELS_DIR     = BASE_DIR / "models"
if MODELS_DIR.exists():
    os.environ.setdefault("OLLAMA_MODELS", str(MODELS_DIR))


# arXiv topics to fetch (https://arxiv.org/category_taxonomy)
ARXIV_TOPICS = [
    "cs.AI",    # AI
    "cs.LG",    # ML
    "cs.CL",    # NLP
]

MAX_PAPERS_PER_TOPIC = 5 

# Chunking
CHUNK_SIZE    = 600    # target characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks


# Embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Ollama
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_EMBED_MODEL  = "nomic-embed-text"        # used if i switch from sentence-transformers
OLLAMA_BASE_URL     = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


# Retrieval 
TOP_K            = 6      # number of chunks returned per query
SCORE_THRESHOLD  = 0.40   # minimum cosine similarity to keep a FAISS result
RRF_K            = 60     # RRF constant higher = softer rank penalty


# Streamlit UI 
APP_TITLE    = "Research Paper Assistant"
APP_ICON     = "📄"
