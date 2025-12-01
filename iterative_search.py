import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Configuration
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'quietflamingo/dnabert2-no-flashattention'
CHROMA_DB_PATH = 'chroma_db'
COLLECTION_NAME = 'dna_sequences'
MAX_LEN = 256 
K_NEIGHBORS = 10 # User mentioned "top 10"
TEST_SAMPLE_SIZE = 10
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

# Function to generate embeddings
def get_embeddings(sequences):
    sequences = [s.replace('\n', '').strip() for s in sequences]
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs[0][:, 0, :].cpu().numpy()
    return embeddings

# Re-load Test Data (Same logic as test_search.py)
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

# Iterative Search Logic
def iterative_search(sequence, collection, k=K_NEIGHBORS):
    # Generate embedding
    embedding = get_embeddings([sequence])[0].tolist()
    
    current_filter = {"split": "train"}
    predicted_taxonomy = {}
    
    print(f"\n--- Starting Iterative Search ---")
    
    for level in HIERARCHY:
        # Construct ChromaDB where clause
        if len(current_filter) > 1:
            where_clause = {"$and": [{k: v} for k, v in current_filter.items()]}
        else:
            where_clause = current_filter

        # Query ChromaDB with current filter
        results = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where_clause
        )
        
        if not results['metadatas'] or not results['metadatas'][0]:
            print(f"No results found at level {level}. Stopping.")
            break
            
        metadatas = results['metadatas'][0]
        
        # Extract values for the current level
        values = [m.get(level, "Unknown") for m in metadatas]
        
        # Count occurrences
        counts = Counter(values)
        most_common_val, count = counts.most_common(1)[0]
        confidence = count / len(values)
        
        print(f"Level: {level}, Top value: {most_common_val}, Confidence: {confidence:.2f} ({count}/{len(values)})")
        
        # Update filter and prediction regardless of confidence
        if most_common_val != "Unknown":
            current_filter[level] = most_common_val
            predicted_taxonomy[level] = {
                "value": most_common_val,
                "confidence": confidence
            }
            print(f"-> Locking {level} = {most_common_val} (Confidence: {confidence:.2f})")
        else:
            print(f"-> Unknown value. Cannot refine further.")
            break
            
    # Calculate overall confidence score
    # Weighted average where higher levels (Kingdom) have more weight? 
    # Or just a product of confidences? 
    # The user asked for "taking into account the decreasing importance of each step in the hierarchy"
    # This implies Kingdom is MORE important than Species, or vice versa?
    # Usually, getting Kingdom right is "easier" and less specific. Getting Species right is hard.
    # If we want a single score, maybe a weighted sum of confidences?
    # Let's implement a simple weighted score where Kingdom has weight 1, Phylum 1, etc.
    # Or maybe decreasing weights? "decreasing importance of each step" -> Kingdom is most important?
    # Let's assume Kingdom (index 0) is most important, Species (index 6) is least important for the "base" classification,
    # BUT usually in taxonomy, specific identification is the goal.
    # However, the prompt says "decreasing importance of each step". 
    # Let's interpret "decreasing importance" as: Kingdom (high weight) -> Species (low weight).
    # Weights: Kingdom=7, Phylum=6, ..., Species=1.
    
    total_weight = 0
    weighted_confidence_sum = 0
    
    weights = {
        'Kingdom': 7, 'Phylum': 6, 'Class': 5, 
        'Order': 4, 'Family': 3, 'Genus': 2, 'Species': 1
    }
    
    for level, data in predicted_taxonomy.items():
        w = weights.get(level, 1)
        weighted_confidence_sum += data['confidence'] * w
        total_weight += w
        
    overall_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0
    
    return predicted_taxonomy, overall_confidence

# Run on a few samples
print(f"Running iterative search on {TEST_SAMPLE_SIZE} samples...")
test_subset = test_df.head(TEST_SAMPLE_SIZE)

for idx, row in test_subset.iterrows():
    print(f"\nQuery Sequence ID: {row.name}") # Assuming index is meaningful or just use row index
    true_tax = {level: row[level] for level in HIERARCHY}
    print(f"True Taxonomy: {true_tax}")
    
    prediction, overall_score = iterative_search(str(row['Sequence']), collection)
    
    print(f"Final Prediction: {prediction}")
    print(f"Overall Weighted Confidence: {overall_score:.4f}")
    
    # Check correctness
    correct_levels = 0
    for level, data in prediction.items():
        if str(true_tax.get(level)) == data['value']:
            correct_levels += 1
    print(f"Correct Levels: {correct_levels}/{len(prediction)} (of predicted)")

