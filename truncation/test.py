import pandas as pd
from Bio.Seq import Seq

df = pd.read_parquet("coi_validated_df_leray.parquet")

fwd = "GGTCAACAAATCATAAAGATATTGG"
rev = "TAAACTTCAGGGTGACCAAAAAATCA"
rev_rc = str(Seq(rev).reverse_complement())

stats = {
    "both_primers": 0,
    "only_forward": 0,
    "only_reverse_rc": 0,
    "neither": 0
}

for seq in df["Sequence"].astype(str).str.upper():
    has_fwd = fwd in seq
    has_revrc = rev_rc in seq

    if has_fwd and has_revrc:
        stats["both_primers"] += 1
    elif has_fwd:
        stats["only_forward"] += 1
    elif has_revrc:
        stats["only_reverse_rc"] += 1
    else:
        stats["neither"] += 1

total = len(df)

print("\n--- PRIMER PRESENCE DIAGNOSTICS ---")
for k, v in stats.items():
    print(f"{k:15s}: {v:7d} ({100*v/total:.2f}%)")
