import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import time

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
    # Usamos analyzer='char' y ngram_range=(5, 5) para k-mers normales de longitud 5
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(5, 5), lowercase=False)),
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
