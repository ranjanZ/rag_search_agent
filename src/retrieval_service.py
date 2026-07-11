import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.config import (FAISS_INDEX_PATH, FAISS_MAPPING_PATH, BM25_INDEX_PATH, NAME_BM25_INDEX_PATH,
                    EMBEDDING_MODEL_NAME, TOP_K_SEMANTIC, TOP_K_LEXICAL, 
                    FINAL_TOP_K, RRF_K)

# --- Score Thresholds ---
SEMANTIC_THRESHOLD = 0.3   # Minimum Cosine Similarity for FAISS
LEXICAL_THRESHOLD = 0.01   # Minimum BM25 Score for Lexical Search

class RetrievalEngine:
    def __init__(self):
        print("Loading Retrieval Engine...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.load_indexes()
        print("✅ Retrieval Engine Ready.")

    def load_indexes(self):
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_MAPPING_PATH, 'rb') as f:
            self.faiss_chunks = pickle.load(f)
            
        with open(BM25_INDEX_PATH, 'rb') as f:
            self.bm25 = pickle.load(f)
            
        with open(NAME_BM25_INDEX_PATH, 'rb') as f:
            self.name_bm25 = pickle.load(f)

    def semantic_search(self, query):
        query_vec = self.model.encode([query])
        faiss.normalize_L2(query_vec)
        scores, indices = self.faiss_index.search(query_vec, TOP_K_SEMANTIC)
        
        valid_indices = []
        valid_scores = []
        for score, idx in zip(scores[0], indices[0]):
            # Filter out FAISS padding (-1) and apply semantic threshold
            if idx != -1 and score >= SEMANTIC_THRESHOLD:
                valid_indices.append(int(idx))
                valid_scores.append(float(score))
                
        return valid_indices, valid_scores

    def lexical_search(self, query):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:TOP_K_LEXICAL]
        top_scores = scores[top_indices]
        
        valid_indices = []
        valid_scores = []
        for idx, score in zip(top_indices, top_scores):
            # Apply lexical threshold
            if score >= LEXICAL_THRESHOLD:
                valid_indices.append(int(idx))
                valid_scores.append(float(score))
                
        return valid_indices, valid_scores

    def name_lexical_search(self, query):
        tokenized_query = query.lower().split()
        scores = self.name_bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:TOP_K_LEXICAL]
        top_scores = scores[top_indices]
        
        valid_indices = []
        valid_scores = []
        for idx, score in zip(top_indices, top_scores):
            # Apply lexical threshold
            if score >= LEXICAL_THRESHOLD:
                valid_indices.append(int(idx))
                valid_scores.append(float(score))
                
        return valid_indices, valid_scores

    def reciprocal_rank_fusion(self, semantic_ids, lexical_ids, name_lexical_ids=None):
        scores = {}
        
        for rank, idx in enumerate(semantic_ids):
            scores[idx] = scores.get(idx, 0) + (1.0 / (RRF_K + rank + 1))
            
        for rank, idx in enumerate(lexical_ids):
            scores[idx] = scores.get(idx, 0) + (1.0 / (RRF_K + rank + 1))
            
        if name_lexical_ids:
            for rank, idx in enumerate(name_lexical_ids):
                scores[idx] = scores.get(idx, 0) + (1.5 / (RRF_K + rank + 1))
            
        sorted_indices = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return sorted_indices[:FINAL_TOP_K]

    def hybrid_search(self, query):
        # 1. Get filtered IDs and Scores from all 3 retrievers
        sem_ids, sem_scores = self.semantic_search(query)
        lex_ids, lex_scores = self.lexical_search(query)
        lex_ids_name, lex_scores_name = self.name_lexical_search(query)
        
        # 2. Perform Reciprocal Rank Fusion
        fused_ids = self.reciprocal_rank_fusion(sem_ids, lex_ids, lex_ids_name)
        
        # 3. Retrieve full chunk data
        results = [self.faiss_chunks[idx] for idx in fused_ids]
        
        # 4. Build the results dictionary
        results_dict = {
            "fused": results,
            "semantic": {
                "ids": sem_ids,
                "scores": sem_scores,
                "chunks": [self.faiss_chunks[idx] for idx in sem_ids]
            },
            "lexical": {
                "ids": lex_ids,
                "scores": lex_scores,
                "chunks": [self.faiss_chunks[idx] for idx in lex_ids]
            },
            "name_lexical": {
                "ids": lex_ids_name,
                "scores": lex_scores_name,
                "chunks": [self.faiss_chunks[idx] for idx in lex_ids_name]
            }
        }
        return results_dict

# Singleton instance for the app
engine = RetrievalEngine()

def retrieve_context(query: str):
    results = engine.hybrid_search(query)
    return results["fused"]  # Return only the fused results for the chat service


if __name__ == "__main__":
    # Test the retrieval engine
    test_query = "Prof Salem mail id"
    print(f"🔎 Testing query: '{test_query}'\n")
    print(f"📏 Thresholds -> Semantic: {SEMANTIC_THRESHOLD} | Lexical: {LEXICAL_THRESHOLD}\n")
    
    results_dict = engine.hybrid_search(test_query)
    
    print("\n=== RETRIEVER METRICS ===")
    for retriever_name, data in results_dict.items():
        if retriever_name == "fused":
            continue
        print(f"\n[{retriever_name.upper()}]")
        for idx, score in zip(data['ids'], data['scores']):
            print(f"Chunk ID: {idx:>3} | Score: {score:.4f}")


    # ==========================================================
    # 1. TEST THE SEMANTIC RETRIEVER
    # ==========================================================
    print("="*70)
    print("🧠 SEMANTIC RETRIEVER RESULTS (FAISS Cosine Similarity)")
    print("="*70)
    
    sem_data = results_dict['semantic']
    if not sem_data['ids']:
        print("⚠️ No semantic results passed the threshold.")
    else:
        for i, (idx, score) in enumerate(zip(sem_data['ids'], sem_data['scores'])):
            chunk = sem_data['chunks'][i]
            print(f"\n--- Semantic Result {i+1} ---")
            print(f"Chunk ID    : {chunk['chunk_id']}")
            print(f"Document ID : {chunk['metadata'].get('document_id', 'N/A')}")
            print(f"Score       : {score:.4f} (Cosine Similarity)")
            print(f"Metadata    : {chunk['metadata']}")
            print(f"Text        : {chunk['enriched_text'][:250]}...")
            print("-" * 70)

    # ==========================================================
    # 2. TEST THE LEXICAL RETRIEVER
    # ==========================================================
    print("\n" + "="*70)
    print("📚 LEXICAL RETRIEVER RESULTS (BM25)")
    print("="*70)
    
    lex_data = results_dict['lexical']
    if not lex_data['ids']:
        print("⚠️ No lexical results passed the threshold.")
    else:
        for i, (idx, score) in enumerate(zip(lex_data['ids'], lex_data['scores'])):
            chunk = lex_data['chunks'][i]
            print(f"\n--- Lexical Result {i+1} ---")
            print(f"Chunk ID    : {chunk['chunk_id']}")
            print(f"Document ID : {chunk['metadata'].get('document_id', 'N/A')}")
            print(f"Score       : {score:.4f}")
            print(f"Text        : {chunk['enriched_text'][:150]}...")
            print("-" * 70)

    # ==========================================================
    # 3. PRINT THE FINAL FUSED RESULTS
    # ==========================================================
    print("\n" + "="*70)
    print("🏆 FINAL FUSED RESULTS (Reciprocal Rank Fusion)")
    print("="*70)
    
    results = results_dict['fused']
    if not results:
        print("⚠️ No final results passed the thresholds.")
    else:
        for i, res in enumerate(results):
            print(f"\n--- Fused Result {i+1} ---")
            print(f"Chunk ID    : {res['chunk_id']}")
            print(f"Document ID : {res['metadata'].get('document_id', 'N/A')}")
            print(f"Metadata    : {res['metadata']}")
            print(f"Text        : {res['enriched_text'][:250]}...")
            print("-" * 70)