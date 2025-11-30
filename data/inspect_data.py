import pandas as pd

try:
    df = pd.read_csv('data/final_dataset.csv', nrows=5)
    print(df.head())
    print(df.columns)
except Exception as e:
    print(f"Error reading csv: {e}")
