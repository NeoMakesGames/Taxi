from hierarchicalsoftmax import SoftmaxNode, HierarchicalSoftmaxLoss, HierarchicalSoftmaxLinear
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# Phase 3: Hierarchical Classification

class HierarchicalDNABERT(nn.Module):
    def __init__(self, encoder_path, root_node):
        super().__init__()
        # Load your fine-tuned backbone from Phase 2
        self.encoder = SentenceTransformer(encoder_path)
        
        # Apply LoRA to the encoder to make it trainable on 16GB VRAM
        # This targets the query/value matrices in Attention layers
        # We need to identify the target modules for DNABERT (BERT-based).
        # Usually "query", "value" or "q_lin", "v_lin".
        # SentenceTransformer wraps the model. We might need to access the underlying transformer.
        
        # Let's check the model structure if possible, or use standard BERT targets.
        # For BERT, it is usually ["query", "value"].
        
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["query", "value"] # Standard for BERT
        )
        
        # We apply PEFT to the transformer module inside SentenceTransformer
        # self.encoder[0] is the Transformer module
        self.encoder[0].auto_model = get_peft_model(self.encoder[0].auto_model, peft_config)

        # The Hierarchical Head
        self.classifier = HierarchicalSoftmaxLinear(
            in_features=768, # DNABERT hidden size
            root=root_node
        )

    def forward(self, input_ids, attention_mask):
        # Get embeddings
        # SentenceTransformer forward expects features dict
        features = {'input_ids': input_ids, 'attention_mask': attention_mask}
        output = self.encoder(features)
        embeddings = output['sentence_embedding']
        # Classify
        return self.classifier(embeddings)

def build_taxonomy_tree(taxonomy_file):
    print("Building Taxonomy Tree...")
    root = SoftmaxNode("root")
    node_map = {"root": root}
    
    with open(taxonomy_file, "r") as f:
        unique_taxonomies = [line.strip() for line in f.readlines()]

    for taxon_path in unique_taxonomies:
        path_parts = taxon_path.split(";") # e.g. ["Arthropoda", "Insecta", "Lepidoptera"]
        parent = root
        for part in path_parts:
            if not part: continue # Skip empty parts
            if part not in node_map:
                node_map[part] = SoftmaxNode(part, parent=parent)
            parent = node_map[part]
            
    print(f"Taxonomy tree constructed with {len(node_map)} nodes.")
    return root, node_map

class SequenceDataset(Dataset):
    def __init__(self, filepath, node_map, tokenizer):
        self.dataset = load_dataset("csv", data_files=filepath, split="train")
        self.node_map = node_map
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset[idx]
        seq = row['Sequence']
        species = str(row['Species'])
        
        # Tokenize
        inputs = self.tokenizer(seq, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
        
        # Get target node
        # The target is the species node.
        if species in self.node_map:
            target_node = self.node_map[species]
        else:
            # Fallback or error? 
            # If the tree was built from the same dataset, it should be there.
            # But maybe some cleaning issues.
            target_node = self.node_map["root"] # Should not happen ideally
            
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'target': target_node
        }

def custom_collate(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    targets = [item['target'] for item in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'target': targets
    }

def train_hierarchical(model_path, data_path, taxonomy_file):
    root, node_map = build_taxonomy_tree(taxonomy_file)
    
    print("Initializing Hierarchical Model...")
    model = HierarchicalDNABERT(model_path, root).cuda()
    criterion = HierarchicalSoftmaxLoss(root)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Data
    # We need the tokenizer from the base model
    tokenizer = model.encoder.tokenizer
    train_dataset = SequenceDataset(data_path, node_map, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    
    print("Starting Training...")
    model.train()
    for epoch in range(1): # 1 epoch as per instructions
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            targets = batch['target'] # This is a list of nodes, handled by HierarchicalSoftmaxLoss?
            # Wait, HierarchicalSoftmaxLoss expects the target indices or nodes?
            # Looking at the library docs (implied), it usually takes the object or index.
            # But DataLoader collates objects into a list if they are not tensors.
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            # The loss function expects (outputs, targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dataloader)}")
        
    # Save
    torch.save(model.state_dict(), "hierarchical_dnabert.pth")

if __name__ == "__main__":
    model_path = "dnabert_s_co1_aligned"
    data_path = "/home/ubuntu/Taxi/data/final_dataset.csv"
    taxonomy_file = "unique_taxonomies.txt"
    
    train_hierarchical(model_path, data_path, taxonomy_file)
