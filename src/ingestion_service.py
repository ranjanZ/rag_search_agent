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
from typing import Dict, Any, List, Optional

# Import from updated config
from src.config import (FAISS_INDEX_PATH, FAISS_MAPPING_PATH, BM25_INDEX_PATH, NAME_BM25_INDEX_PATH,
                    EMBEDDING_MODEL_NAME, DATA_DIR, BASE_DIR, INDEX_DIR,
                    DATASET_CONFIG, DEFAULT_DATASETS, QA_CHUNK_MIN_LENGTH, QA_CHUNK_MAX_LENGTH)


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



def chunk_qa_text(question: str, context: str, doc_id: str, 
                  is_answerable: Optional[bool] = None,
                  answer_text: Optional[str] = None,
                  min_length: int = QA_CHUNK_MIN_LENGTH,
                  max_length: int = QA_CHUNK_MAX_LENGTH) -> List[Dict[str, Any]]:
    """
    Chunks QA pairs (question, context) into retrievable format.
    For long contexts, splits into multiple chunks while keeping the question.
    
    Args:
        question: The question text
        context: The context/passage text
        doc_id: Unique document ID
        is_answerable: Whether the question has an answer (for SQuAD v2 style)
        answer_text: The answer text if available
        min_length: Minimum chunk length
        max_length: Maximum chunk length
    
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    # Clean inputs
    question = clean_text(question)
    context = clean_text(context)
    
    if not question or not context:
        return []
    
    # Split context into sentences for fine-grained chunking
    sentences = re.split(r'(?<=[.!?]) +', context)
    
    current_chunk = ""
    for sent in sentences:
        # Check if adding this sentence exceeds max_length
        if len(current_chunk) + len(sent) + 1 <= max_length:
            current_chunk += (" " if current_chunk else "") + sent
        else:
            # Save current chunk if it meets minimum length
            if len(current_chunk) >= min_length:
                enriched_text = f"Context: {current_chunk}"
                metadata = {
                    "doc_type": "qa_pair",
                    "document_id": doc_id,
                    "question": question,
                    "is_answerable": is_answerable,
                    "answer_text": answer_text if is_answerable else None
                }
                chunks.append({
                    "text": current_chunk,
                    "enriched_text": enriched_text,
                    "metadata": metadata
                })
            # Start new chunk with current sentence
            current_chunk = sent
    
    # Don't forget the last chunk
    if len(current_chunk) >= min_length:
        enriched_text = f"Context: {current_chunk}"
        metadata = {
            "doc_type": "qa_pair",
            "document_id": doc_id,
            "question": question,
            "is_answerable": is_answerable,
            "answer_text": answer_text if is_answerable else None
        }
        chunks.append({
            "text": current_chunk,
            "enriched_text": enriched_text,
            "metadata": metadata
        })
    
    # Fallback: if context was too short, create one chunk anyway
    if not chunks and context:
        enriched_text = f"Context: {context}"
        metadata = {
            "doc_type": "qa_pair",
            "document_id": doc_id,
            "question": question,
            "is_answerable": is_answerable,
            "answer_text": answer_text if is_answerable else None
        }
        chunks.append({
            "text": context,
            "enriched_text": enriched_text,
            "metadata": metadata
        })
    
    return chunks

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

# --- HuggingFace Dataset Parsers ---

def parse_squad_v2(dataset_name: str = "rajpurkar/squad_v2", 
                   split: str = "validation",
                   sample_size: Optional[int] = None,
                   random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Parser for SQuAD v2 dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'rajpurkar/squad_v2')
        split: Dataset split ('train' or 'validation')
        sample_size: Number of samples to use (None for all)
        random_state: Random seed for reproducibility
    
    Returns:
        List of chunk dictionaries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 'datasets' library not installed. Run: pip install datasets")
        return []
    
    print(f"Loading {dataset_name} (split={split})...")
    dataset = load_dataset(dataset_name, split=split)
    
    # Sample if requested
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=random_state).select(range(sample_size))
        print(f"Sampled {sample_size} examples from {dataset_name}")
    
    chunks = []
    for i, example in enumerate(dataset):
        doc_id = example.get('id', f"squad_{i}")
        question = clean_text(example.get('question', ''))
        context = clean_text(example.get('context', ''))
        
        # SQuAD v2: answers can be empty for unanswerable questions
        answers = example.get('answers', {})
        answer_text_list = answers.get('text', []) if isinstance(answers, dict) else []
        is_answerable = len(answer_text_list) > 0
        answer_text = answer_text_list[0] if is_answerable else None
        
        # Create chunks using the QA chunking function
        qa_chunks = chunk_qa_text(
            question=question,
            context=context,
            doc_id=str(doc_id),
            is_answerable=is_answerable,
            answer_text=answer_text
        )
        chunks.extend(qa_chunks)
    
    print(f"✅ Extracted {len(chunks)} chunks from {dataset_name}")
    return chunks


def parse_newsqa(dataset_name: str = "news_qa",
                 split: str = "validation",
                 sample_size: Optional[int] = None,
                 random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Parser for NewsQA dataset from HuggingFace.
    
    NewsQA has a unique structure: each example contains a paragraph with multiple
    question-answer pairs. We need to iterate through the questions and answers lists.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        sample_size: Number of samples to use
        random_state: Random seed for reproducibility
    
    Returns:
        List of chunk dictionaries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 'datasets' library not installed. Run: pip install datasets")
        return []
    
    print(f"Loading {dataset_name} (split={split})...")
    dataset = load_dataset(dataset_name, split=split)
    
    # Sample if requested
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=random_state).select(range(sample_size))
        print(f"Sampled {sample_size} examples from {dataset_name}")
    
    chunks = []
    total_qa_pairs = 0
    
    for i, example in enumerate(dataset):
        # NewsQA structure: paragraph contains the story, questions/answers are lists
        context = clean_text(example.get('paragraph', ''))
        questions = example.get('questions', [])
        answers = example.get('answers', [])
        
        # Skip if no valid context
        if not context:
            continue
        
        # Iterate through each question-answer pair in this paragraph
        for j, (question, answer) in enumerate(zip(questions, answers)):
            question = clean_text(question)
            
            # Handle answer format - NewsQA answers can be empty strings for unanswerable
            if isinstance(answer, str):
                answer_text = clean_text(answer)
                is_answerable = len(answer_text) > 0
            elif isinstance(answer, dict):
                answer_text_list = answer.get('text', [])
                if isinstance(answer_text_list, list) and len(answer_text_list) > 0:
                    answer_text = clean_text(answer_text_list[0])
                    is_answerable = len(answer_text) > 0
                else:
                    answer_text = None
                    is_answerable = False
            else:
                answer_text = None
                is_answerable = False
            
            # Skip if no valid question
            if not question:
                continue
            
            # Create unique doc_id for each Q&A pair
            doc_id = f"newsqa_{i}_q{j}"
            total_qa_pairs += 1
            
            # Create chunks using the QA chunking function
            qa_chunks = chunk_qa_text(
                question=question,
                context=context,
                doc_id=str(doc_id),
                is_answerable=is_answerable,
                answer_text=answer_text if is_answerable else None
            )
            chunks.extend(qa_chunks)
    
    print(f"✅ Extracted {len(chunks)} chunks from {dataset_name} ({total_qa_pairs} Q&A pairs)")
    return chunks


def load_huggingface_dataset(dataset_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generic loader for HuggingFace datasets based on config.
    
    Args:
        dataset_config: Configuration dictionary with keys:
            - type: 'huggingface'
            - name: dataset name (e.g., 'rajpurkar/squad_v2')
            - split: dataset split
            - sample_size: optional sampling size
            - random_state: random seed
    
    Returns:
        List of chunk dictionaries
    """
    dataset_name = dataset_config.get('name', '')
    split = dataset_config.get('split', 'validation')
    sample_size = dataset_config.get('sample_size')
    random_state = dataset_config.get('random_state', 42)
    
    # Route to specific parser based on dataset name
    if 'squad' in dataset_name.lower():
        return parse_squad_v2(dataset_name, split, sample_size, random_state)
    elif 'news' in dataset_name.lower():
        return parse_newsqa(dataset_name, split, sample_size, random_state)
    else:
        # Generic handler for other QA datasets
        try:
            from datasets import load_dataset
        except ImportError:
            print("❌ 'datasets' library not installed. Run: pip install datasets")
            return []
        
        print(f"Loading {dataset_name} (split={split})...")
        dataset = load_dataset(dataset_name, split=split)
        
        if sample_size and len(dataset) > sample_size:
            dataset = dataset.shuffle(seed=random_state).select(range(sample_size))
            print(f"Sampled {sample_size} examples from {dataset_name}")
        
        chunks = []
        for i, example in enumerate(dataset):
            # Try common field names for QA datasets
            doc_id = example.get('id', f"hf_{i}")
            question = clean_text(example.get('question', '') or example.get('q', ''))
            context = clean_text(example.get('context', '') or example.get('passage', '') or example.get('document', ''))
            
            if not question or not context:
                continue
            
            # Try to extract answer info
            answers = example.get('answers', {})
            if isinstance(answers, dict):
                answer_text_list = answers.get('text', [])
            elif isinstance(answers, list):
                answer_text_list = answers
            else:
                answer_text_list = []
            
            is_answerable = len(answer_text_list) > 0 if answer_text_list else False
            answer_text = answer_text_list[0] if is_answerable else None
            
            qa_chunks = chunk_qa_text(
                question=question,
                context=context,
                doc_id=str(doc_id),
                is_answerable=is_answerable,
                answer_text=answer_text
            )
            chunks.extend(qa_chunks)
        
        print(f"✅ Extracted {len(chunks)} chunks from {dataset_name}")
        return chunks


# --- Main Ingestion Pipeline ---
def load_all_data(datasets_to_load: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load data from configured datasets.
    
    Args:
        datasets_to_load: List of dataset names to load from DATASET_CONFIG.
                         If None, uses DEFAULT_DATASETS.
    
    Returns:
        List of all chunks from all datasets
    """
    if datasets_to_load is None:
        datasets_to_load = DEFAULT_DATASETS
    
    print(f"Loading datasets: {datasets_to_load}")
    all_chunks = []
    chunk_id = 0
    
    for dataset_name in datasets_to_load:
        if dataset_name not in DATASET_CONFIG:
            print(f"⚠️ Dataset '{dataset_name}' not found in DATASET_CONFIG. Skipping.")
            continue
        
        config = DATASET_CONFIG[dataset_name]
        dataset_type = config.get('type', 'local_json')
        
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name} (type: {dataset_type})")
        print('='*60)
        
        chunks = []
        
        if dataset_type == 'huggingface':
            chunks = load_huggingface_dataset(config)
        elif dataset_type == 'local_json':
            filepath = Path(config.get('path', ''))
            if filepath.exists():
                # Use custom parser if specified, otherwise use generic
                parser_name = config.get('parser')
                if parser_name == 'parse_faculty':
                    chunks = parse_faculty(filepath)
                elif parser_name == 'parse_research_projects':
                    chunks = parse_research_projects(filepath)
                else:
                    chunks = parse_generic_json(filepath)
            else:
                print(f"❌ File not found: {filepath}")
                continue
        elif dataset_type == 'local_csv':
            filepath = Path(config.get('path', ''))
            if filepath.exists():
                chunks = parse_generic_csv(filepath)
            else:
                print(f"❌ File not found: {filepath}")
                continue
        else:
            print(f"⚠️ Unknown dataset type: {dataset_type}. Skipping.")
            continue
        
        # Assign unique sequential IDs to all chunks
        for chunk in chunks:
            chunk["chunk_id"] = chunk_id
            chunk_id += 1
        
        all_chunks.extend(chunks)
        print(f"  ✅ Total chunks from {dataset_name}: {len(chunks)}")
    
    print(f"\n{'='*60}")
    print(f"🎉 Total chunks across all datasets: {len(all_chunks)}")
    print('='*60)
    
    return all_chunks


# Legacy function for backward compatibility
def load_all_data_legacy():
    """Legacy function that loads from DATA_DIR (backward compatibility)."""
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
            chunk["chunk_id"] = chunk_id
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

def run_ingestion(datasets_to_load: Optional[List[str]] = None):
    """
    Run the full ingestion pipeline.
    
    Args:
        datasets_to_load: Optional list of dataset names to load.
                         If None, uses DEFAULT_DATASETS from config.
    """
    chunks = load_all_data(datasets_to_load)
    if not chunks:
        print("No data to ingest.")
        return
        
    print(f"\n🎉 Total chunks created across all sources: {len(chunks)}")
    
    build_bm25_index(chunks)
    build_faiss_index(chunks)
    build_name_index(chunks)
    
    print("🚀 Ingestion pipeline complete!")

if __name__ == "__main__":
    import sys
    
    # Allow passing dataset names as command line arguments
    # Example: python ingestion_service.py squad_v2 newsqa
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
        print(f"Loading datasets from command line: {datasets}")
        run_ingestion(datasets_to_load=datasets)
    else:
        run_ingestion()
