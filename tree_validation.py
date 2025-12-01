import pandas as pd
import numpy as np
import joblib
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from taxonomic_tree import TaxonomicNode, ConstantModel, predict_single, HIERARCHY

def evaluate_model(root, X_test, df_test):
    print("\nEvaluando rendimiento del modelo en el conjunto de prueba...")
    start_time = time.time()
    
    total_samples = X_test.shape[0]
    
    predictions = []
    print(f"Prediciendo en {total_samples} muestras...")
    for i in range(total_samples):
        if i % 1000 == 0 and i > 0:
            print(f"Procesadas {i}/{total_samples} muestras...")
        x_sample = X_test[i]
        pred = predict_single(root, x_sample)
        predictions.append(pred)
        
    print(f"\nEstadísticas de Rendimiento (Conjunto de Prueba: {total_samples} muestras):")
    print(f"{'Nivel':<15} | {'Exactitud':<10} | {'Puntaje F1':<10} | {'Correctos':<10}")
    print("-" * 60)
    
    for level in HIERARCHY:
        y_true = []
        y_pred = []
        correct_count = 0
        
        for i, pred in enumerate(predictions):
            actual = df_test.iloc[i]
            
            # True label
            y_true.append(actual[level])
            
            # Predicted label
            if level in pred:
                y_pred.append(pred[level])
                if pred[level] == actual[level]:
                    correct_count += 1
            else:
                y_pred.append("Unpredicted") # Treat missing prediction as a distinct class
        
        acc = accuracy_score(y_true, y_pred)
        # Use weighted F1 score to account for class imbalance
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"{level:<15} | {acc:.2%}     | {f1:.4f}     | {correct_count}/{total_samples}")
    
    print(f"\nTiempo de evaluación: {time.time() - start_time:.2f}s")

def main():
    # We need to reproduce the data loading and splitting exactly as in taxonomic_tree.py
    DATASET_PERCENTAGE = 0.1
    
    print("Cargando datos para validación...")
    try:
        df = pd.read_csv('data/final_dataset.csv')
        if DATASET_PERCENTAGE < 1.0:
            df = df.sample(frac=DATASET_PERCENTAGE, random_state=42)
    except FileNotFoundError:
        print("Datos no encontrados.")
        return

    # Clean data
    df = df.dropna(subset=['Sequence'] + HIERARCHY)
    print(f"Datos cargados: {len(df)} muestras")

    # Vectorize
    print("Vectorizando...")
    # Note: We must use the same vectorizer parameters
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5), lowercase=False, max_features=5000)
    X = vectorizer.fit_transform(df['Sequence'])
    
    # Split train/test
    indices = np.arange(len(df))
    _, X_test, _, y_test_idx = train_test_split(X, indices, test_size=0.2, random_state=42)
    
    df_test = df.iloc[y_test_idx]

    print("Cargando modelo...")
    try:
        data = joblib.load('taxonomic_tree.pkl')
        root = data['root']
    except FileNotFoundError:
        print("Archivo del modelo 'taxonomic_tree.pkl' no encontrado. Por favor ejecute taxonomic_tree.py primero.")
        return

    evaluate_model(root, X_test, df_test)

if __name__ == "__main__":
    main()
