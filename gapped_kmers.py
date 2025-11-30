import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import time

# Función para generar k-mers con huecos
def get_gapped_kmers(sequence, gap_mask='11011'):
    """
    Genera k-mers con huecos para una secuencia dada basada en una máscara.
    mask: cadena de '1's y '0's. '1' significa incluir, '0' significa hueco.
    Ejemplo: '11011' extrae las posiciones 0, 1, 3, 4.
    """
    kmers = []
    mask_len = len(gap_mask)
    if len(sequence) < mask_len:
        return []
    
    indices = [i for i, char in enumerate(gap_mask) if char == '1']
    
    # Optimization: List comprehension is faster than loop with append
    # But for very large strings, this is still slow in pure Python.
    # For a prototype, this is fine.
    
    # Pre-calculate slices for vectorization if possible, but strings are immutable.
    # Let's stick to simple iteration for clarity first.
    
    for i in range(len(sequence) - mask_len + 1):
        # Extract characters where mask is '1'
        # subseq = sequence[i : i + mask_len]
        # gapped_kmer = "".join([subseq[j] for j in indices])
        
        # Slightly faster: direct access
        gapped_kmer = "".join([sequence[i+j] for j in indices])
        kmers.append(gapped_kmer)
        
    return kmers

# Envoltorio para CountVectorizer
def gapped_kmer_tokenizer(text):
    # Using a fixed mask for this example: 11011 (gapped 4-mer)
    return get_gapped_kmers(text, gap_mask='11011')

def to_gapped_kmers_string(seq):
    return " ".join(get_gapped_kmers(seq, gap_mask='11011'))

def main():
    print("Cargando datos...")
    start_time = time.time()
    try:
        # Loading a subset for demonstration if the file is huge
        # df = pd.read_csv('data/final_dataset.csv', nrows=10000) 
        # Let's try loading all, but if it fails or is slow, we can adjust.
        df = pd.read_csv('data/final_dataset.csv')
    except FileNotFoundError:
        print("Archivo de datos no encontrado en data/final_dataset.csv")
        return

    print(f"Datos cargados: {df.shape} filas. Tiempo: {time.time() - start_time:.2f}s")
    
    # Target: 'Phylum'
    target_col = 'Phylum'
    if target_col not in df.columns:
        print(f"Columna {target_col} no encontrada. Disponibles: {df.columns}")
        return

    # Drop NA in Sequence or Target
    df = df.dropna(subset=['Sequence', target_col])
    
    # Filter out classes with very few samples to avoid split errors
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts > 5].index
    df = df[df[target_col].isin(valid_classes)]
    
    X = df['Sequence']
    y = df[target_col]
    
    print(f"Clases objetivo ({len(y.unique())}): {y.unique()}")
    
    # Split data
    print("Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Vectorizando y entrenando (esto puede tardar un poco)...")
    
    # Pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=gapped_kmer_tokenizer, token_pattern=None, lowercase=False)),
        ('classifier', LogisticRegression(max_iter=100, n_jobs=-1, verbose=1))
    ])
    
    train_start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Entrenamiento completado en {time.time() - train_start:.2f}s")
    
    print("Evaluando...")
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Precisión: {acc:.4f}")
    
    # Print classification report
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
