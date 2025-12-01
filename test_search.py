import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'quietflamingo/dnabert2-no-flashattention'
CHROMA_DB_PATH = 'chroma_db'
COLLECTION_NAME = 'dna_sequences'
MAX_LEN = 256 
K_NEIGHBORS = 5
TEST_SAMPLE_SIZE = 100 # Number of test samples to evaluate

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
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Connected to collection '{COLLECTION_NAME}' with {count} documents.")
        if count == 0:
            print("Warning: Collection is empty. Please run embed_sequences.py first.")
            exit(1)
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' not found. Please run embed_sequences.py first.")
        exit(1)
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

# Re-load Test Data
print(f"Reloading dataset to reconstruct test split...")
if not os.path.exists(CSV_PATH):
    print(f"File not found: {CSV_PATH}")
    exit(1)

chunks = []
read_chunk_size = 50000 
try:
    # We must use the exact same logic as embed_sequences.py to get the same split
    for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size), desc="Reading and sampling"):
        sampled_chunk = chunk.sample(frac=0.1, random_state=42)
        chunks.append(sampled_chunk)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

df_sample = pd.concat(chunks)
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
print(f"Total Test set size: {len(test_df)}")

# Perform Search
print(f"Running search on a subset of {TEST_SAMPLE_SIZE} test samples...")
test_subset = test_df.head(TEST_SAMPLE_SIZE)

results = []

for idx, row in tqdm(test_subset.iterrows(), total=len(test_subset), desc="Searching"):
    sequence = str(row['Sequence'])
    true_metadata = {
        'Kingdom': row['Kingdom'],
        'Phylum': row['Phylum'],
        'Class': row['Class'],
        'Order': row['Order'],
        'Family': row['Family'],
        'Genus': row['Genus'],
        'Species': row['Species']
    }
    
    # Generate embedding
    try:
        embedding = get_embeddings([sequence])[0]
    except Exception as e:
        print(f"Error embedding sequence: {e}")
        continue
        
    # Query ChromaDB
    # Filter by split='train' to ensure we retrieve neighbors from the training set
    query_result = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=K_NEIGHBORS,
        where={"split": "train"} 
    )
    
    # Analyze results
    if not query_result['metadatas'] or not query_result['metadatas'][0]:
        continue
        
    # Get the top match (nearest neighbor)
    top_match = query_result['metadatas'][0][0]
    distance = query_result['distances'][0][0]
    
    match_correct = {}
    for level in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']:
        # Handle NaN/None
        true_val = str(true_metadata[level]) if pd.notna(true_metadata[level]) else "Unknown"
        pred_val = top_match.get(level, "Unknown")
        match_correct[level] = (true_val == pred_val)
    
    results.append(match_correct)

# Calculate Accuracy
if results:
    df_results = pd.DataFrame(results)
    print("\n--- Search Accuracy (Top-1 Neighbor from Train Set) ---")
    print(f"Evaluated on {len(results)} samples.")
    for col in df_results.columns:
        acc = df_results[col].mean()
        print(f"{col}: {acc:.2%}")
else:
    print("No results generated.")
