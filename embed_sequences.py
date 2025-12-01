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
BATCH_SIZE = 16  # Conservative batch size to avoid OOM
MAX_LEN = 256 

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
print("Initializing ChromaDB...")
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # Use Hierarchical Navigable Small World (HNSW) with Cosine similarity
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    exit(1)

# Function to generate embeddings
def get_embeddings(sequences):
    # Clean sequences: remove newlines and whitespace
    sequences = [s.replace('\n', '').strip() for s in sequences]
    
    # Tokenize
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding (first token)
    # The user's example used: cls_output = outputs[0][:, 0, :]
    embeddings = outputs[0][:, 0, :].cpu().numpy()
    return embeddings

# Process CSV in chunks
print(f"Processing {CSV_PATH}...")

# Check if file exists
if not os.path.exists(CSV_PATH):
    print(f"File not found: {CSV_PATH}")
    exit(1)

# 1. Load and Sample 10%
print("Sampling 10% of the dataset...")
chunks = []
# Use a larger chunksize for reading to speed up I/O, we will sample down immediately
read_chunk_size = 50000 
try:
    for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size), desc="Reading and sampling"):
        sampled_chunk = chunk.sample(frac=0.1, random_state=42)
        chunks.append(sampled_chunk)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

df_sample = pd.concat(chunks)
print(f"Sampled {len(df_sample)} sequences.")

# 2. Train/Test Split
print("Performing Train/Test Split (80/20)...")
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
train_df['split'] = 'train'
test_df['split'] = 'test'

# Recombine to process in one go
processing_df = pd.concat([train_df, test_df])

# 3. Process in batches
total_rows = len(processing_df)
num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Starting embedding generation for {total_rows} sequences in {num_batches} batches...")

for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Embedding batches"):
    batch_df = processing_df.iloc[i : i + BATCH_SIZE]
    
    try:
        # Clean sequences
        sequences = batch_df['Sequence'].astype(str).tolist()
        headers = batch_df['Header'].astype(str).tolist()
        
        # Prepare metadata
        metadatas = []
        for _, row in batch_df.iterrows():
            # Handle NaN values
            meta = {
                'Kingdom': str(row['Kingdom']) if pd.notna(row['Kingdom']) else "Unknown",
                'Phylum': str(row['Phylum']) if pd.notna(row['Phylum']) else "Unknown",
                'Class': str(row['Class']) if pd.notna(row['Class']) else "Unknown",
                'Order': str(row['Order']) if pd.notna(row['Order']) else "Unknown",
                'Family': str(row['Family']) if pd.notna(row['Family']) else "Unknown",
                'Genus': str(row['Genus']) if pd.notna(row['Genus']) else "Unknown",
                'Species': str(row['Species']) if pd.notna(row['Species']) else "Unknown",
                'split': row['split']
            }
            metadatas.append(meta)
            
        # Generate embeddings
        embeddings = get_embeddings(sequences)
        
        # Add to ChromaDB
        collection.upsert(
            ids=headers,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=sequences
        )
        
    except Exception as e:
        print(f"Error processing batch starting at index {i}: {e}")
        continue

print("Done! Embeddings stored in ChromaDB.")
