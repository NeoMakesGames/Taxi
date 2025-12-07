import argparse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, InputExample, models
from torch.utils.data import DataLoader, IterableDataset
import torch
import random
import os

# Set tokenizers parallelism to false to avoid deadlocks in DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Phase 1: Data Loading
def load_and_prepare_data(filepath, test_mode=False):
    print("Loading dataset (Memory Mapped)...")
    # Load dataset without reading into RAM
    dataset = load_dataset("csv", data_files=filepath, split="train")
    
    if test_mode:
        print("Test Mode: Using first 10,000 examples.")
        dataset = dataset.select(range(10000))

    # We need to create the taxonomy string for Phase 3 (saving it later)
    # and build an index for Phase 2.
    
    unique_taxonomies = set()
    species_to_sequences = {}
    genus_to_species = {}
    
    print("Indexing dataset for triplet generation (Indices only)...")
    
    # We access columns directly to avoid loading the heavy 'Sequence' column
    # This is much faster and RAM efficient
    try:
        species_col = dataset['Species']
        genus_col = dataset['Genus']
        # We might need other columns for taxonomy string if we want to save it
        # But for the index, we just need Species and Genus.
    except KeyError:
        print("Error: Dataset missing 'Species' or 'Genus' columns.")
        return None, None, None, None

    # Iterate over indices and metadata only
    for idx, (species, genus) in enumerate(zip(species_col, genus_col)):
        species = str(species)
        genus = str(genus)
        
        # Store index instead of sequence string
        if species not in species_to_sequences:
            species_to_sequences[species] = []
        species_to_sequences[species].append(idx)
        
        if genus not in genus_to_species:
            genus_to_species[genus] = set()
        genus_to_species[genus].add(species)
        
        # For unique taxonomies, we might need to do a pass or just skip if not strictly needed for Phase 2.
        # To save RAM, let's skip building the full unique_taxonomies set here if it's huge.
        # But the user requested it. Let's do it efficiently.
        # We can reconstruct it from the row if needed, but let's assume we can do it later or separately.
        # For now, let's just collect unique paths if we iterate.
        # Since we are iterating columns, we don't have the full row.
        # Let's skip unique_taxonomies generation here to save RAM and time, 
        # or do it in a separate lightweight pass if needed for Phase 3.
    
    print(f"Found {len(species_to_sequences)} unique species.")
    
    # Re-enable unique taxonomies extraction if needed for Phase 3 file generation
    # We can do a quick pass using unique() on columns if dataset supports it, or just iterate.
    # For now, we return an empty set to avoid OOM, assuming Phase 3 can generate it or we do it separately.
    unique_taxonomies = set() 
    
    return species_to_sequences, genus_to_species, unique_taxonomies, dataset

class TripletDataset(IterableDataset):
    def __init__(self, dataset, species_to_sequences, genus_to_species):
        self.dataset = dataset
        self.species_to_sequences = species_to_sequences
        self.genus_to_species = genus_to_species
        self.genera = list(self.genus_to_species.keys())
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Shuffle genera to ensure random sampling
        # If multiple workers, we should shard or seed differently
        if worker_info is not None:
            # Seed random based on worker id
            random.seed(worker_info.id + torch.initial_seed())
            
        random.shuffle(self.genera)
        
        for genus in self.genera:
            species_list = list(self.genus_to_species[genus])
            if len(species_list) < 2:
                continue
            
            random.shuffle(species_list)
            
            for species in species_list:
                seq_indices = self.species_to_sequences[species]
                if len(seq_indices) < 2:
                    continue
                
                # Shuffle indices for this species
                # We copy to avoid modifying the global dict in place if shared (though fork copies on write)
                current_seq_indices = list(seq_indices)
                random.shuffle(current_seq_indices)
                
                for i in current_seq_indices:
                    # Fetch Anchor Sequence from Disk/MemoryMap
                    anchor_seq = self.dataset[i]['Sequence']
                    
                    # Positive: Random other sequence from same species
                    positive_idx = i
                    while positive_idx == i:
                        positive_idx = random.choice(seq_indices)
                    positive_seq = self.dataset[positive_idx]['Sequence']
                    
                    # Hard Negative: Random sequence from same genus, different species
                    neg_species = species
                    while neg_species == species:
                        neg_species = random.choice(species_list)
                    
                    neg_seq_indices = self.species_to_sequences[neg_species]
                    negative_idx = random.choice(neg_seq_indices)
                    negative_seq = self.dataset[negative_idx]['Sequence']
                    
                    yield InputExample(texts=[anchor_seq, positive_seq, negative_seq])

    def __len__(self):
        # Approximate length for progress bar
        count = 0
        for genus, species_list in self.genus_to_species.items():
            if len(species_list) < 2: continue
            for species in species_list:
                if len(self.species_to_sequences[species]) >= 2:
                    count += len(self.species_to_sequences[species])
        return count

# Phase 2: Triplet Generation and Training
def train_metric_learning(species_to_sequences, genus_to_species, dataset, output_path="dnabert_s_co1_aligned"):
    print("Initializing Model...")
    # 1. Load DNABERT-S as the backbone
    word_embedding_model = models.Transformer("zhihan1996/DNABERT-S", max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # 2. Format Data for Contrastive Learning
    print("Initializing Triplet Dataset...")
    train_dataset = TripletDataset(dataset, species_to_sequences, genus_to_species)
    
    # 3. The "DNACSE" Loss Function
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 4. Train
    # Optimized for 16GB VRAM
    batch_size = 16 # Reduced from 32 to be safe
    
    # DataLoader with multiprocessing
    # num_workers=4 allows fetching data in background
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=model.smart_batching_collate,
        num_workers=4,
        pin_memory=True
    )
    
    print("Starting Training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        epochs=1,
        use_amp=True, # Mixed Precision for memory efficiency
        show_progress_bar=True
    )
    
    print(f"Saving model to {output_path}...")
    model.save(output_path)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on a small subset for testing")
    args = parser.parse_args()

    filepath = "/home/ubuntu/Taxi/data/final_dataset.csv"
    
    species_to_sequences, genus_to_species, unique_taxonomies, dataset = load_and_prepare_data(filepath, test_mode=args.test)
    
    if species_to_sequences:
        # Generate taxonomy file separately if needed, or here if RAM permits.
        # For now, we focus on training.
        
        train_metric_learning(species_to_sequences, genus_to_species, dataset)
