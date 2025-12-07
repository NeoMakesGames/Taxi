import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
from tqdm import tqdm

# Phase 4: Open-Set Discovery

def build_index(model_path, data_path):
    print("Loading Model...")
    model = SentenceTransformer(model_path)
    model.eval()
    
    print("Loading Data...")
    dataset = load_dataset("csv", data_files=data_path, split="train")
    
    print("Computing Embeddings...")
    # We can use model.encode with batching
    # But for 1.7M, we should be careful with RAM if we return all at once.
    # model.encode returns numpy array by default.
    # 1.7M * 768 * 4 bytes = ~5.2 GB. It fits in RAM.
    
    sequences = dataset['Sequence']
    embeddings = model.encode(sequences, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    
    print("Normalizing Embeddings...")
    faiss.normalize_L2(embeddings)
    
    print("Building FAISS Index...")
    # IndexFlatIP is Inner Product, which is Cosine Similarity after normalization
    index = faiss.IndexFlatIP(768) 
    index.add(embeddings)
    
    print(f"Index built with {index.ntotal} vectors.")
    faiss.write_index(index, "co1_taxonomy.index")
    
    return index, embeddings

def discovery_mode(index, model, new_sequence, threshold_known=0.98, threshold_genus=0.90):
    # Get Embedding
    embedding = model.encode([new_sequence], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    
    # Search
    D, I = index.search(embedding, k=1)
    similarity = D[0][0]
    nearest_idx = I[0][0]
    
    print(f"Similarity: {similarity}")
    
    if similarity > threshold_known:
        return "Known Species"
    elif similarity > threshold_genus:
        return "New Species, Known Genus"
    else:
        return "New Genus / Outlier"

if __name__ == "__main__":
    model_path = "dnabert_s_co1_aligned"
    data_path = "/home/ubuntu/Taxi/data/final_dataset.csv"
    
    # Build Index
    index, embeddings = build_index(model_path, data_path)
    
    # Example Usage
    # Load model again for inference (or pass it)
    model = SentenceTransformer(model_path)
    
    # Test with a sequence from the dataset (should be Known Species)
    test_seq = "GATATTGGTACTTTATATTTAATATTCGCAGGGTTTGCTGGTATTATAGGTACAATCTTTTCTGTTGTGATTCGTATGGAATTAGCTTATCCAGGAGATCAAATTTTACAAGGAGATTATCAATTATATAATGTAATTATTACTGCACATGCTTTTATAATGATTTTCTTTATGCTAATGCCTGCTTTAATTGGT"
    result = discovery_mode(index, model, test_seq)
    print(f"Test Result: {result}")
