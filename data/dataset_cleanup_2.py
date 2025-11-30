import pandas as pd

# Leer el conjunto de datos limpio
df = pd.read_csv('data/cleaned_dataset.csv')

# Eliminar filas duplicadas (manteniendo la primera aparición), ignorando la columna 'Header'
subset_columns = [col for col in df.columns if col != 'Header']
df_deduplicated = df.drop_duplicates(subset=subset_columns)

# Guardar el resultado de nuevo en cleaned_dataset.csv
df_deduplicated.to_csv('data/cleaned_dataset.csv', index=False)

print(f"Se eliminaron {len(df) - len(df_deduplicated)} filas duplicadas.")
