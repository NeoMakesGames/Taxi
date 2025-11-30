import pandas as pd

# Read the cleaned dataset
df = pd.read_csv('data/cleaned_dataset.csv')

# Remove duplicate rows (keeping the first occurrence), ignoring the 'Header' column
subset_columns = [col for col in df.columns if col != 'Header']
df_deduplicated = df.drop_duplicates(subset=subset_columns)

# Save the result back to cleaned_dataset.csv
df_deduplicated.to_csv('data/cleaned_dataset.csv', index=False)

print(f"Removed {len(df) - len(df_deduplicated)} duplicate rows.")
