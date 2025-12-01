import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# Configuration
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'quietflamingo/dnabert2-no-flashattention'
CHROMA_DB_PATH = 'chroma_db'
COLLECTION_NAME = 'dna_sequences'
MAX_LEN = 256 
K_NEIGHBORS = 10 
HIERARCHY = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

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

# Initialize ChromaDB
print("Connecting to ChromaDB...")
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Connected to collection '{COLLECTION_NAME}' with {collection.count()} documents.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    exit(1)

# Re-load Test Data
print(f"Reloading dataset to reconstruct test split...")
if not os.path.exists(CSV_PATH):
    print(f"File not found: {CSV_PATH}")
    exit(1)

chunks = []
read_chunk_size = 50000 
try:
    for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size), desc="Reading and sampling"):
        sampled_chunk = chunk.sample(frac=0.1, random_state=42)
        chunks.append(sampled_chunk)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

df_sample = pd.concat(chunks)
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
print(f"Total Test set size: {len(test_df)}")

# Limit to 300 samples for testing
test_df = test_df.head(300)
print(f"Limiting evaluation to {len(test_df)} samples.")

# Function to generate embeddings in batches
def get_embeddings_batched(sequences, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Generating Embeddings"):
        batch = sequences[i:i+batch_size]
        batch = [s.replace('\n', '').strip() for s in batch]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs[0][:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Iterative Search Logic (Optimized)
def iterative_search_from_embedding(embedding, collection, k=K_NEIGHBORS):
    current_filter = {"split": "train"}
    predicted_taxonomy = {}
    
    for level in HIERARCHY:
        # Construct ChromaDB where clause
        if len(current_filter) > 1:
            where_clause = {"$and": [{k: v} for k, v in current_filter.items()]}
        else:
            where_clause = current_filter

        # Query ChromaDB with current filter
        try:
            results = collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=k,
                where=where_clause,
                include=['metadatas'] # Optimization: Only fetch metadata
            )
        except Exception as e:
            break
        
        if not results['metadatas'] or not results['metadatas'][0]:
            break
            
        metadatas = results['metadatas'][0]
        values = [m.get(level, "Unknown") for m in metadatas]
        
        if not values:
            break

        counts = Counter(values)
        most_common_val, count = counts.most_common(1)[0]
        confidence = count / len(values)
        
        if most_common_val != "Unknown":
            current_filter[level] = most_common_val
            predicted_taxonomy[level] = {
                "value": most_common_val,
                "confidence": confidence
            }
        else:
            break
            
    # Calculate overall confidence score
    total_weight = 0
    weighted_confidence_sum = 0
    weights = {'Kingdom': 7, 'Phylum': 6, 'Class': 5, 'Order': 4, 'Family': 3, 'Genus': 2, 'Species': 1}
    
    for level, data in predicted_taxonomy.items():
        w = weights.get(level, 1)
        weighted_confidence_sum += data['confidence'] * w
        total_weight += w
        
    overall_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0
    
    return predicted_taxonomy, overall_confidence

# Evaluation Loop
print(f"Starting evaluation on {len(test_df)} samples...")

# Pre-compute embeddings
print("Pre-computing embeddings for test set...")
test_sequences = test_df['Sequence'].astype(str).tolist()
test_embeddings = get_embeddings_batched(test_sequences, batch_size=64)

y_true = {level: [] for level in HIERARCHY}
y_pred = {level: [] for level in HIERARCHY}
confidences = []

print("Running iterative search...")
for idx, (embedding, (_, row)) in tqdm(enumerate(zip(test_embeddings, test_df.iterrows())), total=len(test_df), desc="Evaluating"):
    true_tax = {level: str(row[level]) if pd.notna(row[level]) else "Unknown" for level in HIERARCHY}
    
    prediction, overall_score = iterative_search_from_embedding(embedding, collection)
    confidences.append(overall_score)
    
    for level in HIERARCHY:
        y_true[level].append(true_tax[level])
        if level in prediction:
            y_pred[level].append(prediction[level]['value'])
        else:
            y_pred[level].append("Unpredicted")


# Calculate Metrics
print("\n--- Evaluation Results ---")
metrics = []

for level in HIERARCHY:
    # Filter out 'Unknown' in ground truth if desired? 
    # Usually we want to evaluate against known ground truth.
    # But here we keep everything.
    
    acc = accuracy_score(y_true[level], y_pred[level])
    
    # Weighted average for multi-class
    prec = precision_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    rec = recall_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    f1 = f1_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    
    metrics.append({
        "Level": level,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

df_metrics = pd.DataFrame(metrics)
print(df_metrics.to_string(index=False))

print(f"\nAverage Overall Confidence: {np.mean(confidences):.4f}")

# Save results to CSV
df_metrics.to_csv("iterative_search_results.csv", index=False)
print("Results saved to iterative_search_results.csv")
