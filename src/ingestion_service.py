import os
import json
import csv
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re

from config import (FAISS_INDEX_PATH, FAISS_MAPPING_PATH, BM25_INDEX_PATH, NAME_BM25_INDEX_PATH,
                    EMBEDDING_MODEL_NAME,DATA_DIR,BASE_DIR,INDEX_DIR)

# --- Path Configuration ---
# # Dynamically finds the root directory whether run from root or inside src/
# SCRIPT_DIR = Path(__file__).parent.resolve()
# BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'src' else SCRIPT_DIR

# DATA_DIR = BASE_DIR / "data" / "raw"
# INDEX_DIR = BASE_DIR / "indexes"
# INDEX_DIR.mkdir(parents=True, exist_ok=True)


# --- Models ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Helper Functions ---
def clean_text(text):
    """Basic text cleaning."""
    if not text: return ""
    return str(text).strip()

def chunk_text(text, min_length=50):
    """Splits text into paragraphs/chunks."""
    if not text: return []
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs and len(text) > min_length:
        return [text]
    return [p for p in paragraphs if len(p) >= min_length]

# --- Specific Parsers ---
def parse_research_projects(filepath):
    """Parser specifically tuned for the research_projects.json schema."""
    print(f"  Parsing research projects from {filepath.name}...")
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        projects = json.load(f)
        
    for project in projects:
        # Extract the document ID from the JSON
        doc_id = project.get("id")
        
        title = clean_text(project.get("Title"))
        owner = clean_text(project.get("Owner"))
        category = clean_text(project.get("Category"))
        description = clean_text(project.get("Description"))
        
        enriched_prefix = f"Research Project: {title}. Category: {category}. Lead: {owner}. "
        
        paragraphs = chunk_text(description)
        if not paragraphs:
            paragraphs = [description] if description else []
            
        for para in paragraphs:
            chunks.append({
                "text": para,
                "enriched_text": enriched_prefix + para,
                "metadata": {
                    "source_file": filepath.name,
                    "doc_type": "research_project",
                    "document_id": doc_id,  # Added document_id
                    "title": title,
                    "owner": owner,
                    "category": category
                }
            })
    return chunks

##########faculty 

def chunk_text_robust(text, max_chars=800, min_chars=20):
    """
    Robustly chunks text. Splits by newlines, then by sentences if a block is too long.
    Hard splits if a single sentence exceeds max_chars.
    800 chars is roughly 200 tokens, ideal for all-MiniLM-L6-v2.
    """
    if not text:
        return []
    
    # 1. Split by newlines first
    raw_blocks = [b.strip() for b in text.split('\n') if b.strip()]
    if not raw_blocks:
        raw_blocks = [text.strip()]
        
    chunks = []
    for block in raw_blocks:
        if len(block) <= max_chars:
            if len(block) >= min_chars:
                chunks.append(block)
        else:
            # 2. Block is too long, split by sentences
            sentences = re.split(r'(?<=[.!?]) +', block)
            temp_chunk = ""
            for sent in sentences:
                if len(temp_chunk) + len(sent) + 1 <= max_chars:
                    temp_chunk += (" " if temp_chunk else "") + sent
                else:
                    if len(temp_chunk) >= min_chars:
                        chunks.append(temp_chunk)
                    
                    # 3. If the single sentence is still too long, hard split it
                    while len(sent) > max_chars:
                        chunks.append(sent[:max_chars])
                        sent = sent[max_chars:]
                    temp_chunk = sent
                    
            if len(temp_chunk) >= min_chars:
                chunks.append(temp_chunk)
            elif temp_chunk: 
                # Append remaining small fragment to the last chunk if it exists
                if chunks:
                    chunks[-1] += " " + temp_chunk
                else:
                    chunks.append(temp_chunk)
                
    return chunks


def parse_faculty(filepath):
    """Parser specifically tuned for the exact all_faculty.json schema."""
    print(f"  Parsing faculty data from {filepath.name}...")
    all_chunks = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        faculty_list = json.load(f)
        
    for person in faculty_list:
        # Extract the document ID from the JSON
        doc_id = person.get("id")
        
        # 1. Extract core metadata using exact top-level keys
        name = clean_text(person.get("name"))
        title = clean_text(person.get("title"))
        email = clean_text(person.get("email"))
        url = clean_text(person.get("url"))
        
        # Skip if there's no name
        if not name:
            continue
            
        # 2. Extract the 'tabs' dictionary
        tabs = person.get("tabs", {})
        if not isinstance(tabs, dict):
            tabs = {} # Fallback if 'tabs' is missing or malformed
            
        # 3. Extract specific sections using exact keys inside 'tabs'
        sections_to_process = {
            "biography": clean_text(tabs.get("biography")),
            "education": clean_text(tabs.get("education")),
            "research": clean_text(tabs.get("research")),
            "accolades": clean_text(tabs.get("accolades")),
            "publications": clean_text(tabs.get("publications"))
        }
        
        person_chunks = []
        
        # Base enrichment string containing all core metadata
        base_meta_str = f"Faculty Member: {name}. Title: {title}. Email: {email}. URL: {url}."
        
        # 4. Process each section
        for section_name, section_text in sections_to_process.items():
            if not section_text:
                continue
                
            # Use robust chunking to handle long texts (e.g., massive accolades or research lists)
            section_chunks = chunk_text_robust(section_text, max_chars=800, min_chars=20)
            
            # Prefix the chunk with metadata and the specific section name
            section_prefix = f"{base_meta_str} Section: {section_name.capitalize()}. "
            
            for chunk in section_chunks:
                person_chunks.append({
                    "text": chunk,
                    "enriched_text": section_prefix + chunk,
                    "metadata": {
                        "source_file": filepath.name,
                        "doc_type": "faculty",
                        "document_id": doc_id,  # Added document_id
                        "name": name,
                        "title": title,
                        "email": email,
                        "url": url,
                        "section": section_name # Tracks if this chunk is from bio, education, etc.
                    }
                })
                
        # 5. Fallback if absolutely no sections were found for this person
        if not person_chunks:
            summary = f"{name} is a {title} at MBZUAI."
            person_chunks.append({
                "text": summary,
                "enriched_text": f"{base_meta_str} Summary: {summary}",
                "metadata": {
                    "source_file": filepath.name,
                    "doc_type": "faculty",
                    "document_id": doc_id,  # Added document_id
                    "name": name,
                    "title": title,
                    "email": email,
                    "url": url,
                    "section": "summary"
                }
            })
            
        all_chunks.extend(person_chunks)
        
    return all_chunks

# --- Generic Parsers (Fallback for unknown files) ---
def parse_generic_csv(filepath):
    """Generic parser for any CSV file. Combines all columns into text."""
    print(f"  Parsing generic CSV from {filepath.name}...")
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        for row in reader:
            # Extract document ID if it exists in the CSV
            doc_id = row.get("id")
            
            text_parts = []
            metadata = {
                "source_file": filepath.name, 
                "doc_type": "csv_row",
                "document_id": doc_id  # Added document_id
            }
            
            for header in headers:
                val = clean_text(row.get(header))
                if val:
                    text_parts.append(f"{header}: {val}")
                    metadata[header.lower().replace(" ", "_")] = val
                    
            full_text = ". ".join(text_parts)
            if len(full_text) < 20: continue
                
            first_col_val = clean_text(row.get(headers[0])) if headers else ""
            enriched_prefix = f"Record: {first_col_val}. " if first_col_val else ""
            
            chunks.append({
                "text": full_text,
                "enriched_text": enriched_prefix + full_text,
                "metadata": metadata
            })
    return chunks

def parse_generic_json(filepath):
    """Generic parser for any JSON file containing a list of dictionaries."""
    print(f"  Parsing generic JSON from {filepath.name}...")
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Extract document ID if it exists in the JSON
                doc_id = item.get("id")
                
                text_parts = [f"{k}: {clean_text(v)}" for k, v in item.items() if clean_text(v)]
                full_text = ". ".join(text_parts)
                if len(full_text) < 20: continue
                
                metadata = {
                    "source_file": filepath.name, 
                    "doc_type": "json_item",
                    "document_id": doc_id  # Added document_id
                }
                metadata.update({k.lower().replace(" ", "_"): clean_text(v) for k, v in item.items() if clean_text(v)})
                
                first_val = clean_text(list(item.values())[0]) if item else ""
                enriched_prefix = f"Record: {first_val}. " if first_val else ""
                
                chunks.append({
                    "text": full_text,
                    "enriched_text": enriched_prefix + full_text,
                    "metadata": metadata
                })
    return chunks

# --- Main Ingestion Pipeline ---
def load_all_data():
    print(f"Scanning for data files in {DATA_DIR}...")
    all_chunks = []
    chunk_id = 0
    
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return []

    # Map specific filenames to their dedicated parsers
    specific_parsers = {
        "research_projects.json": parse_research_projects,
        "all_faculty.json": parse_faculty,
        # You can easily add more specific mappings here in the future
    }
    
    # Find all JSON and CSV files in the raw data directory
    files_to_process = list(DATA_DIR.glob("*.json")) + list(DATA_DIR.glob("*.csv"))
    
    if not files_to_process:
        print("❌ No JSON or CSV files found in data/raw/")
        return []

    for filepath in files_to_process:
        print(f"Processing {filepath.name}...")
        chunks = []
        
        # Route to specific parser if defined, otherwise use generic fallback
        if filepath.name in specific_parsers:
            chunks = specific_parsers[filepath.name](filepath)
            print(f"*********{filepath}")
        elif filepath.suffix == ".json":
            chunks = parse_generic_json(filepath)
        elif filepath.suffix == ".csv":
            chunks = parse_generic_csv(filepath)
            
        # Assign unique sequential IDs to all chunks as chunk_id
        for chunk in chunks:
            chunk["chunk_id"] = chunk_id  # Changed from "id" to "chunk_id"
            chunk_id += 1
            
        all_chunks.extend(chunks)
        print(f"  ✅ Extracted {len(chunks)} chunks.")

    return all_chunks

def build_bm25_index(chunks):
    print("\nBuilding BM25 Lexical Index...")
    corpus = [chunk["enriched_text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25, f)
    print("✅ BM25 Index saved.")

def build_faiss_index(chunks):
    print("Building FAISS Semantic Index...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    texts = [chunk["enriched_text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_MAPPING_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    print("✅ FAISS Index saved.")

def build_name_index(chunks):
    """
    Builds a specialized BM25 index ONLY for names, owners, and titles.
    This ensures searching "Karray" or "Song" isn't penalized by long publication texts.
    
    KEY IMPROVEMENT: Stores names BOTH as full phrases AND as individual words
    to handle partial name queries like "Prof Salam" -> "Salam"
    """
    print("\nBuilding Specialized Name/Entity Index...")
    identity_keys = {"name", "owner", "title", "category", "department"}
    corpus = []
    
    for chunk in chunks:
        meta = chunk["metadata"]
        identity_words = []
        
        for key in identity_keys:
            if key in meta and meta[key]:
                # Clean punctuation (e.g., turns "Fakhreddine (Fakhri) Karray" into "fakhreddine fakhri karray")
                clean_val = re.sub(r'[^\w\s]', '', str(meta[key])).lower()
                
                # Split into individual words for partial matching
                # e.g., "salem lahlou" -> ["salem", "lahlou", "salem lahlou"]
                words = clean_val.split()
                
                # Add individual words (for partial name matches like "Salam")
                for word in words:
                    if len(word) > 1:  # Skip single characters
                        identity_words.append(word)
                
                # Add the full phrase (for exact matches)
                if len(words) > 1:
                    identity_words.append(clean_val)
                
                # BOOST: If it's a name or owner, add individual words twice to increase BM25 Term Frequency weight
                if key in {"name", "owner"}:
                    for word in words:
                        if len(word) > 1:
                            identity_words.append(word)
                    identity_words.append(clean_val)  # Extra boost for full name
                    
        # BM25 crashes if a document is completely empty, so add a dummy token if no identity is found
        if not identity_words:
            identity_words = ["unknown_entity"]
            
        corpus.append(identity_words)

    name_bm25 = BM25Okapi(corpus)
    with open(NAME_BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(name_bm25, f)
    print("✅ Name/Entity Index saved.")

def run_ingestion():
    chunks = load_all_data()
    if not chunks:
        print("No data to ingest.")
        return
        
    print(f"\n🎉 Total chunks created across all sources: {len(chunks)}")
    
    build_bm25_index(chunks)
    build_faiss_index(chunks)
    build_name_index(chunks) # <-- NEW
    
    print("🚀 Ingestion pipeline complete!")

if __name__ == "__main__":
    run_ingestion()