import pandas as pd

def clean_func(string: str) -> str:
    parts = string.split('_')
    
    # Rangos para identificar la parte del prefijo de la cadena
    ranks = {'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'}
    
    # Contar cuántas partes al inicio son rangos
    rank_count = 0
    for part in parts:
        if part.lower() in ranks:
            rank_count += 1
        else:
            break
            
    # Devolver la(s) parte(s) entre los rangos y el ID (última parte)
    # Esto generaliza la devolución de parts[-2] para casos estándar y maneja nombres complejos con guiones bajos
    name_parts = [p.strip() for p in parts[rank_count:-1] if p and p.strip()]
    
    if not name_parts:
        # Alternativa para casos inesperados donde no se encuentra ningún nombre
        if len(parts) >= 2:
             return parts[-2]
        return string

    return ' '.join(name_parts)
    
def clean_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df[column_name] = df[column_name].apply(clean_func)
    return df

df = pd.read_csv('dataset.csv')

# Estamos limpiando las columnas: Kingdom, Phylum, Class, Order, Family, Genus, Species
columns_to_clean = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

for column in columns_to_clean:
    df = clean_dataframe(df, column)

df.to_csv('cleaned_dataset.csv', index=False)