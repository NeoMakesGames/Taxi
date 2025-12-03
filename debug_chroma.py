import chromadb
import numpy as np
import time

CHROMA_DB_PATH = 'chroma_db'
COLLECTION_NAME = 'dna_sequences'

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print(f"Connected. Count: {collection.count()}")

# Generate a random embedding
# Need to know the dimension. Let's peek at one item.
print("Peeking at one item...")
peek = collection.peek(limit=1)
if peek['embeddings'] is None or len(peek['embeddings']) == 0:
    print("No embeddings found in peek!")
    exit(1)

dim = len(peek['embeddings'][0])
print(f"Embedding dimension: {dim}")

random_embedding = np.random.rand(dim).tolist()

print("Querying...")
start = time.time()
results = collection.query(
    query_embeddings=[random_embedding],
    n_results=10,
    include=['metadatas']
)
end = time.time()
print(f"Query done in {end - start:.4f} seconds.")
print(f"Results: {len(results['ids'][0])}")
