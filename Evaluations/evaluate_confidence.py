import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
from tqdm import tqdm
import os
from collections import Counter
from scipy.stats import entropy

# Configuration
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'zhihan1996/DNABERT-S' # Using _s model as per user preference
MILVUS_DB_PATH = 'milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences_s'
MAX_LEN = 256
K_NEIGHBORS_PRED = 1
K_NEIGHBORS_ENTROPY = 10
HIERARCHY = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
OUTPUT_CSV = 'confidence_evaluation_results.csv'

# Initialize Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Model and Tokenizer
print("Loading model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Initialize Milvus
print("Connecting to Milvus...")
try:
    client = MilvusClient(uri=MILVUS_DB_PATH)
except Exception as e:
    print(f"Error connecting to Milvus: {e}")
    exit(1)

def get_embeddings(sequences, batch_size=32):
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        batch = [s.replace('\n', '').strip() for s in batch]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs[0][:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

def calculate_entropy(labels):
    """Calculates Shannon entropy of label distribution."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    probs = [count / len(labels) for count in counts.values()]
    return entropy(probs, base=2)

def search_and_evaluate(embedding, client):
    # Search for K_NEIGHBORS_ENTROPY to get enough data for entropy
    # We will use the first one for prediction/distance
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[embedding.tolist()],
            limit=K_NEIGHBORS_ENTROPY,
            output_fields=HIERARCHY + ["distance"] # Ensure distance is returned if possible, though usually it's in the hit object
        )
    except Exception as e:
        print(f"Error querying Milvus: {e}")
        return None

    if not results or not results[0]:
        return None

    hits = results[0]
    nearest = hits[0]
    
    # Prediction based on nearest (k=1)
    prediction = {level: nearest['entity'].get(level, "Unknown") for level in HIERARCHY}
    
    # Distance to nearest
    # Milvus returns 'distance' in the hit object. 
    # For COSINE metric, distance is usually 1 - cosine_similarity or just cosine distance.
    # If metric_type was COSINE, Milvus returns IP (Inner Product) if vectors are normalized, or Cosine Distance.
    # Let's assume it returns a distance where smaller is closer (or larger is closer depending on metric).
    # In embed_sequences.py, metric_type="COSINE". 
    # Milvus Python SDK usually returns 'distance' field.
    dist = nearest['distance']
    
    # Confidence based on distance (simple inverse or just the raw distance)
    # We'll store the raw distance for now.
    
    # Entropy calculation per level
    entropies = {}
    for level in HIERARCHY:
        # Get labels from all k neighbors
        labels = [hit['entity'].get(level, "Unknown") for hit in hits]
        entropies[level] = calculate_entropy(labels)
        
    return {
        "prediction": prediction,
        "distance": dist,
        "entropies": entropies,
        "ground_truth_neighbors": hits # Optional: store neighbor info
    }

def main():
    # Load Test Data
    # We need to replicate the split or load a cache
    # Assuming we can just load the full dataset and split again with same seed
    print("Loading dataset...")
    # For speed, we might want to just sample if the file is huge and we only want to demonstrate
    # But user wants evaluation.
    # Let's try to read a sample first to not OOM
    try:
        # Read full CSV but only keep test part
        # This might be slow. 
        # If test_dataset_cache.csv exists, use it.
        if os.path.exists('data/test_dataset_cache.csv'):
             test_df = pd.read_csv('data/test_dataset_cache.csv')
        else:
             # Fallback: Read full and split
             # This is risky with 1.7M rows on 16GB RAM if we load all into pandas.
             # But 1.7M rows with text might take ~1-2GB RAM. It should be fine.
             df = pd.read_csv(CSV_PATH)
             from sklearn.model_selection import train_test_split
             _, test_df = train_test_split(df, test_size=0.2, random_state=42)
             
        print(f"Test set size: {len(test_df)}")
        
        # Limit for demonstration if needed, but user asked for evaluation.
        # We'll process a subset (e.g. 1000) to be quick, or all if user wants.
        # User said "dataset is large", so maybe just 1000 samples for now to test the script?
        # I'll set a limit of 1000 for this run to ensure it finishes.
        test_df = test_df.head(1000)
        print(f"Evaluating on first {len(test_df)} samples...")
        
        results_data = []
        
        sequences = test_df['Sequence'].astype(str).tolist()
        
        # Batch processing
        batch_size = 32
        for i in tqdm(range(0, len(sequences), batch_size), desc="Evaluating"):
            batch_seqs = sequences[i:i+batch_size]
            batch_indices = test_df.index[i:i+batch_size]
            
            # Generate embeddings
            embeddings = get_embeddings(batch_seqs)
            
            for j, embedding in enumerate(embeddings):
                res = search_and_evaluate(embedding, client)
                if res:
                    # Get Ground Truth
                    idx = batch_indices[j]
                    row = test_df.loc[idx]
                    
                    record = {
                        "Sequence_ID": idx, # Or some ID
                        "Distance_Nearest": res['distance']
                    }
                    
                    # Add Predictions and Entropies
                    for level in HIERARCHY:
                        record[f"Pred_{level}"] = res['prediction'][level]
                        record[f"True_{level}"] = row[level]
                        record[f"Entropy_{level}"] = res['entropies'][level]
                        record[f"Match_{level}"] = (res['prediction'][level] == row[level])
                        
                    results_data.append(record)
                    
        # Save results
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Results saved to {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
