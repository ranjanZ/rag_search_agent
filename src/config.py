import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data" / "raw"
INDEX_DIR = BASE_DIR / "data"/"indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
FAISS_MAPPING_PATH = INDEX_DIR / "faiss_mapping.pkl"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
NAME_BM25_INDEX_PATH = INDEX_DIR / "name_bm25.pkl" # <-- NEW

# --- Models ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval Settings ---
TOP_K_SEMANTIC = 10
TOP_K_LEXICAL = 10
FINAL_TOP_K = 5
RRF_K = 60 

# --- DeepInfra LLM Settings ---
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY", "IfQawFoRjh3GajXUFDWnshQvhTtuGZ73F")
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
