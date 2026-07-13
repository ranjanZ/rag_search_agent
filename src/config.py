import os
from pathlib import Path
from typing import Dict, Any, Optional

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data" / "raw"


# --- Models ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval Settings ---
TOP_K_SEMANTIC = 10
TOP_K_LEXICAL = 10
FINAL_TOP_K = 5
RRF_K = 60 

# --- DeepInfra LLM Settings ---
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY", "Note needed")
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# --- Dataset Configuration ---
# Supported dataset types: 'local_json', 'local_csv', 'huggingface'
# For HuggingFace datasets, use the format: 'namespace/dataset_name'
# 
# To use SQuAD v2 or NewsQA, uncomment and configure the respective entries below.
# You can mix multiple datasets by adding them to DEFAULT_DATASETS.
#
DATASET_CONFIG: Dict[str, Any] = {
    # Local JSON files (current setup)
    "faculty": {
        "type": "local_json",
        "path": str(DATA_DIR / "all_faculty.json"),
        "parser": "parse_faculty",
    },
    "research_projects": {
        "type": "local_json",
        "path": str(DATA_DIR / "research_projects.json"),
        "parser": "parse_research_projects",
    },
    
     #SQuAD v2 from HuggingFace - UNCOMMENT TO USE
    "squad_v2": {
         "type": "huggingface",
        "name": "rajpurkar/squad_v2",
         "split": "validation",  # or "train"
         "sample_size": 1000,    # Number of samples to use (None for all)
         "random_state": 42,     # For reproducibility
    },
    
    # NewsQA from HuggingFace - UNCOMMENT TO USE
    "newsqa": {
        "type": "huggingface",
        "name": "gabrieltorresgamez/newsqa",
        "split": "validation",
        "sample_size": 1000,    
        "random_state": 42,
    },
}




DEFAULT_DATASETS = ["squad_v2"]
#DEFAULT_DATASETS = ["newsqa"]
#DEFAULT_DATASETS = ["faculty", "research_projects"]
#DEFAULT_DATASETS = ["faculty", "research_projects", "squad_v2","newsqa"]



# Chunking settings for QA datasets
QA_CHUNK_MIN_LENGTH = 50
QA_CHUNK_MAX_LENGTH = 800

# Domain-specific settings (for off-topic detection)
# Set to None to disable off-topic filtering (useful for general QA datasets)
#DOMAIN_TOPIC = "MBZUAI, artificial intelligence, computer science, research projects, or academic topics"  # Set to None for general QA
DOMAIN_TOPIC=None

dataset_folder_name = "_".join(DEFAULT_DATASETS)
INDEX_DIR = BASE_DIR / "data" / "indexes" / dataset_folder_name
INDEX_DIR.mkdir(parents=True, exist_ok=True)



FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
FAISS_MAPPING_PATH = INDEX_DIR / "faiss_mapping.pkl"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
NAME_BM25_INDEX_PATH = INDEX_DIR / "name_bm25.pkl"
