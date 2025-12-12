import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
import multiprocessing as mp
import os
import time
import numpy as np
import traceback

# -------------------
# User config
# -------------------
INPUT_PARQUET_PATH = 'coi_validated_df_leray.parquet'
OUT_DIR = 'extraction_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

PRIMER_AWARE_CSV = os.path.join(OUT_DIR, 'primer_extracted_sequences.csv')
PRIMER_AWARE_FASTA = os.path.join(OUT_DIR, 'primer_extracted_sequences.fasta')
DNABERT_FASTA = os.path.join(OUT_DIR, 'dnabert_ready_sequences.fasta')
DIAGNOSTICS_CSV = os.path.join(OUT_DIR, 'extraction_diagnostics.csv')

WINDOW_SIZE = 512
WINDOW_STRATEGY = 'center'

NUM_CORES = min(8, os.cpu_count() or 1)
if NUM_CORES < 1:
    NUM_CORES = 1

# Debug prints for tricky sequences (set True temporarily if needed)
DEBUG = False

# -------------------
# Primer sets
# -------------------
PRIMER_SETS = {
    "folmer": {
        "fwd": "GGTCAACAAATCATAAAGATATTGG",
        "rev": "TAAACTTCAGGGTGACCAAAAAATCA",
        "threshold_f": 20.0,
        "threshold_r": 25.0,
        "min_len": 500,
        "max_len": 750
    },
    "leray": {
        "fwd": "GGWACWGGWTGAACWGTWTAYCCYCC",
        "rev": "TAIACYTCIGGRTGICCRAARAAYCA",
        "threshold_f": 18.0,
        "threshold_r": 18.0,
        "min_len": 150,
        "max_len": 700
    }
}

MATCH_SCORE = 2.0
MISMATCH_SCORE = -1.0
GAP_OPEN = -5.0
GAP_EXTEND = -0.5

# -------------------
# IUPAC sanitizer
# -------------------
IUPAC_CHARS = set(list("ACGTURYSWKMBDHVN"))  # uppercase IUPAC; no 'I'
def sanitize_iupac(s: str) -> str:
    """
    Convert input to uppercase and replace any non-IUPAC char (and 'I') with 'N'.
    """
    if s is None:
        return ""
    s_up = str(s).upper()
    out = []
    for ch in s_up:
        if ch == 'I':       # treat inosine as N
            out.append('N')
        elif ch in IUPAC_CHARS:
            out.append(ch)
        else:
            out.append('N')
    return "".join(out)

# -------------------
# SAFE ALIGNER (matrix loaded per worker)
# -------------------
def make_aligner():
    # load matrix inside worker to avoid pickling/import issues
    matrix = substitution_matrices.load("NUC.4.4")
    al = PairwiseAligner()
    al.mode = 'local'
    al.match_score = MATCH_SCORE
    al.mismatch_score = MISMATCH_SCORE
    al.open_gap_score = GAP_OPEN
    al.extend_gap_score = GAP_EXTEND
    al.substitution_matrix = matrix
    return al

# -------------------
def get_best_match_coords(sequence_str, primer_seq, threshold_score, aligner):
    """
    Sanitizes inputs, calls aligner.align with ValueError handling, returns coords or (None,None,0.0).
    """
    seq_s = sanitize_iupac(sequence_str)
    primer_s = sanitize_iupac(primer_seq)

    try:
        alignments = aligner.align(primer_s, seq_s)
    except ValueError as e:
        # e.g. "sequence not in alphabet" — return no hit
        if DEBUG:
            print(f"ALIGN VALUEERROR for primer '{primer_seq}' on seq start '{seq_s[:60]}': {e}")
        return None, None, 0.0
    except Exception as e:
        if DEBUG:
            print(f"ALIGN unexpected error for primer '{primer_seq}': {e}")
        return None, None, 0.0

    if not alignments:
        return None, None, 0.0

    best = max(alignments, key=lambda aln: aln.score)
    score = float(best.score)

    if score < threshold_score:
        return None, None, score

    # cross-version coordinate extraction
    try:
        target_blocks = best.aligned[1]
        if len(target_blocks) == 0:
            return None, None, score
        start = int(target_blocks[0][0])
        end = int(target_blocks[-1][1])
        return start, end, score
    except Exception:
        # fallback older API check
        if hasattr(best, "target_start") and hasattr(best, "target_end"):
            return int(best.target_start), int(best.target_end), score
        return None, None, score

# -------------------
def try_primer_pair(seq_upper, fwd, rev_rc, aligner, thresh_f, thresh_r, min_len, max_len):
    fwd_start, fwd_end, fwd_score = get_best_match_coords(seq_upper, fwd, thresh_f, aligner)
    rev_start, rev_end, rev_score = get_best_match_coords(seq_upper, rev_rc, thresh_r, aligner)

    result = {
        "extracted": None,
        "fwd_score": fwd_score,
        "rev_score": rev_score,
        "fwd_coords": (fwd_start, fwd_end),
        "rev_coords": (rev_start, rev_end)
    }

    if fwd_start is not None and rev_start is not None and fwd_start < rev_end:
        cand = seq_upper[fwd_start:rev_end]
        if min_len <= len(cand) <= max_len:
            result["extracted"] = cand

    return result

# -------------------
def sliding_window_fallback(seq_upper):
    L = len(seq_upper)
    if L >= WINDOW_SIZE:
        start = max(0, (L - WINDOW_SIZE) // 2)
        return seq_upper[start:start + WINDOW_SIZE]
    return seq_upper

# -------------------
def process_chunk(args):
    df_chunk, primer_sets = args
    aligner = make_aligner()

    diagnostics = []
    output_rows = []

    for idx, row in df_chunk.iterrows():
        try:
            header = row.get('Header', f'row_{idx}')
            raw_seq = str(row.get('Sequence', ''))
            seq = sanitize_iupac(raw_seq)

            extracted = None
            primer_family = None

            for fam_name, fam in primer_sets.items():
                # sanitize primers before using RC/alignment
                sanitized_fwd = sanitize_iupac(fam['fwd'])
                sanitized_rev_rc = str(Seq(sanitize_iupac(fam['rev'])).reverse_complement())

                res = try_primer_pair(
                    seq, sanitized_fwd, sanitized_rev_rc, aligner,
                    fam['threshold_f'], fam['threshold_r'],
                    fam['min_len'], fam['max_len']
                )

                if res["extracted"]:
                    extracted = res["extracted"]
                    primer_family = fam_name
                    # capture scores/coords in diagnostics
                    fwd_score = res["fwd_score"]
                    rev_score = res["rev_score"]
                    fwd_coords = res["fwd_coords"]
                    rev_coords = res["rev_coords"]
                    break
                else:
                    fwd_score = res["fwd_score"]
                    rev_score = res["rev_score"]
                    fwd_coords = res["fwd_coords"]
                    rev_coords = res["rev_coords"]

            if extracted:
                dnabert_seq = sliding_window_fallback(extracted)
                extraction_type = "primer"
            else:
                dnabert_seq = sliding_window_fallback(seq)
                extraction_type = "window"

            output_rows.append({
                "Header": header,
                "Extracted_Sequence": extracted,
                "Primer_Family": primer_family,
                "Extraction_Type": extraction_type,
                "DNABERT_Sequence": dnabert_seq,
                "Species": row.get('Species', '')
            })

            diagnostics.append({
                "Header": header,
                "Primer_Family": primer_family,
                "Extraction_Type": extraction_type,
                "fwd_score": fwd_score,
                "rev_score": rev_score,
                "fwd_coords": fwd_coords,
                "rev_coords": rev_coords,
                "input_len": len(seq),
                "extracted_len": len(extracted) if extracted else None
            })

            # optional debugging: show first few problematic sequences
            if DEBUG and (diagnostics and diagnostics[-1]["extracted_len"] is None and len(diagnostics) < 6):
                print(f"DEBUG row {idx}: extraction_type={extraction_type}, fwd_score={fwd_score}, rev_score={rev_score}, header={header}")

        except Exception as e:
            # log and continue; never crash worker
            print(f"WORKER ERROR at {idx}: {e}")
            traceback.print_exc()
            output_rows.append({
                "Header": header,
                "Sequence": extracted if extracted else seq,  # primer-extracted or window fallback
                "Primer_Family": primer_family,
                "Extraction_Type": extraction_type,
                "Kingdom": row.get('Kingdom', ''),
                "Phylum": row.get('Phylum', ''),
                "Class": row.get('Class', ''),
                "Order": row.get('Order', ''),
                "Family": row.get('Family', ''),
                "Genus": row.get('Genus', ''),
                "Species": row.get('Species', '')
                })

            diagnostics.append({
                "Header": header,
                "Primer_Family": None,
                "Extraction_Type": "error",
                "fwd_score": None,
                "rev_score": None,
                "fwd_coords": (None, None),
                "rev_coords": (None, None),
                "input_len": len(seq) if 'seq' in locals() else None,
                "extracted_len": None
            })

    return pd.DataFrame(output_rows), pd.DataFrame(diagnostics)

# -------------------
def run_hybrid_extraction():
    print("Loading input parquet...")
    df = pd.read_parquet(INPUT_PARQUET_PATH)

    chunks = np.array_split(df, NUM_CORES * 4)  # smaller
    args = [(chunk, PRIMER_SETS) for chunk in chunks]

    print(f"Running with {NUM_CORES} workers...")
    start_time = time.time()

    with mp.get_context("spawn").Pool(NUM_CORES) as pool:
        results = pool.map(process_chunk, args)

    # Combine all chunk outputs
    results_df = pd.concat([r[0] for r in results]).reset_index(drop=True)
    diag_df = pd.concat([r[1] for r in results]).reset_index(drop=True)

    # 1️⃣ Save diagnostics as before
    diag_df.to_csv(DIAGNOSTICS_CSV, index=False)

    # 2️⃣ Save full DNABERT-ready CSV with all taxonomy
    results_df.to_csv(os.path.join(OUT_DIR, 'dnabert_ready_full_taxonomy.csv'), index=False)

    # 3️⃣ Optional: keep primer-only CSV as before
    primer_df = results_df[results_df["Extraction_Type"] == "primer"]
    primer_df.to_csv(PRIMER_AWARE_CSV, index=False)


    primer_df = results_df[results_df["Extraction_Type"] == "primer"]
    primer_df.to_csv(PRIMER_AWARE_CSV, index=False)

    records = []
    for _, row in primer_df.iterrows():
        seq_str = row["Extracted_Sequence"]
        if not seq_str:
            continue
        records.append(SeqRecord(
            Seq(seq_str),
            id=str(row["Header"]),
            description=""
        ))

    SeqIO.write(records, PRIMER_AWARE_FASTA, "fasta")

    dna_records = []
    for _, row in results_df.iterrows():
        seq_for_model = row["DNABERT_Sequence"]
        if not seq_for_model:
            continue
        dna_records.append(SeqRecord(
            Seq(seq_for_model),
            id=str(row["Header"]),
            description=""
        ))

    SeqIO.write(dna_records, DNABERT_FASTA, "fasta")

    print("\n✅ DONE")
    print(f"Elapsed time: {time.time() - start_time:.2f}s")

# -------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_hybrid_extraction()
