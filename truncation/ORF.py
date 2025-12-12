import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os

# --- Configuration (ADAPTED FOR LERAY) ---
# New minimum length to safely capture the ~313 bp target
MIN_RAW_SEQUENCE_LENGTH = 250 
GENETIC_CODE = 5 
MIN_PROTEIN_LENGTH = 80 
CHECKPOINT_LEVEL = 'Species'
OUTPUT_PARQUET_PATH = 'coi_validated_df_leray.parquet' 
# -------------------------------------------

def is_functional_coi(sequence: str) -> bool:
    """Checks for a clean, uninterrupted Open Reading Frame (ORF)."""
    sequence = Seq(sequence.upper())
    for frame in range(3):
        try:
            protein_seq = str(sequence[frame:].translate(table=GENETIC_CODE, to_stop=False))
            if '*' not in protein_seq[:-1] and len(protein_seq) >= MIN_PROTEIN_LENGTH:
                return True 
        except Exception:
            pass 
    return False

def filter_data_and_create_checkpoint(input_csv_path, output_fasta_path, checkpoint_csv_path):
    
    print(f"Starting data loading and filtering from: {input_csv_path}...")
    
    df = pd.read_csv(input_csv_path)
    df.columns = ['Header', 'Sequence', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    total_count = len(df)
    
    # 1. IMMEDIATE LENGTH FILTER: Removes sequences shorter than 400 bp
    df['length'] = df['Sequence'].str.len()
    df_length_filtered = df[df['length'] >= MIN_RAW_SEQUENCE_LENGTH].copy()
    skipped_too_short = total_count - len(df_length_filtered)
    
    # 2. ORF FILTER
    print("Applying ORF validation...")
    df_length_filtered['is_functional'] = df_length_filtered['Sequence'].apply(is_functional_coi)
    df_validated = df_length_filtered[df_length_filtered['is_functional']].copy()
    skipped_orf_nuMTs = len(df_length_filtered) - len(df_validated)
    
    # --- Output Generation ---
    
    # A. Write Validated DataFrame to Parquet for use in Step 2
    df_validated.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    
    # B. Write Filtered FASTA file (for general use)
    fasta_records = []
    for _, row in df_validated.iterrows():
        fasta_id = f"{row['Species'].replace(' ', '_')}|{row['Header']}"
        record = SeqRecord(Seq(row['Sequence']), id=fasta_id, description=f"Taxon:{row[CHECKPOINT_LEVEL]}")
        fasta_records.append(record)
        
    SeqIO.write(fasta_records, output_fasta_path, "fasta")
    
    # C. Create CSV Checkpoint
    checkpoint_df = df_validated.groupby(CHECKPOINT_LEVEL)['Header'].count().reset_index()
    checkpoint_df.columns = [CHECKPOINT_LEVEL, 'Sample_Count']
    checkpoint_df.sort_values(by='Sample_Count', ascending=False, inplace=True)
    checkpoint_df.to_csv(checkpoint_csv_path, index=False)
    
    # --- Summary ---
    print("\n--- Filtering & Checkpoint Summary ---")
    print(f"Total sequences processed: {total_count}")
    print(f"Sequences retained (Clean & >= {MIN_RAW_SEQUENCE_LENGTH} bp): {len(df_validated)}")
    print(f"Sequences discarded (Too Short, < {MIN_RAW_SEQUENCE_LENGTH} bp): {skipped_too_short}")
    print(f"Sequences discarded (Likely nuMTs/Errors, ORF failure): {skipped_orf_nuMTs}")
    print(f"Validated DataFrame (full taxonomy) saved to: {OUTPUT_PARQUET_PATH}")
    # ... (rest of summary) ...

# --- Usage ---
filter_data_and_create_checkpoint(
     input_csv_path='dataset_animalia.csv', 
     output_fasta_path='coi_orf_validated_leray.fasta',
     checkpoint_csv_path='coi_class_balance_checkpoint_leray.csv'
 )