import pandas as pd
import numpy as np
import faiss
import pickle
import os
import time
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Configuración
MODEL_DIR = 'faiss_models'
BATCH_SIZE = 1000  # Tamaño del lote para predicción para evitar problemas de memoria

def load_artifacts():
    print("Cargando artefactos...")
    try:
        index = faiss.read_index(os.path.join(MODEL_DIR, 'taxonomy.index'))
        with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        train_df = pd.read_pickle(os.path.join(MODEL_DIR, 'train_data.pkl'))
        test_df = pd.read_pickle(os.path.join(MODEL_DIR, 'test_data.pkl'))
        print("Artefactos cargados exitosamente.")
        return index, vectorizer, train_df, test_df
    except FileNotFoundError as e:
        print(f"Error cargando artefactos: {e}")
        print("Por favor ejecute 'taxonomic_faiss.py' primero para generar los modelos.")
        return None, None, None, None

def predict_taxonomy_batch(index, vectorizer, train_df, query_sequences, k_neighbors=5):
    """
    Predecir taxonomía para una lista de secuencias usando índice Faiss.
    """
    # Vectorizar secuencias de consulta
    query_vectors = vectorizer.transform(query_sequences)
    query_vectors_dense = query_vectors.toarray().astype('float32')
    
    # Buscar en índice Faiss
    D, I = index.search(query_vectors_dense, k_neighbors)
    
    predictions = []
    
    # Niveles taxonómicos a predecir
    levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    for i, neighbors_indices in enumerate(I):
        # Obtener las filas correspondientes a los vecinos más cercanos
        # Usamos iloc para obtener filas por posición entera, lo cual coincide con el índice en Faiss
        neighbors = train_df.iloc[neighbors_indices]
        
        seq_prediction = {}
        for level in levels:
            if level in neighbors.columns:
                # Voto mayoritario para este nivel
                top_class = neighbors[level].mode()
                if not top_class.empty:
                    seq_prediction[level] = top_class[0]
                else:
                    seq_prediction[level] = "Unknown"
        
        predictions.append(seq_prediction)
        
    return pd.DataFrame(predictions)

def evaluate_model(index, vectorizer, train_df, test_df):
    print(f"\nEvaluando en {len(test_df)} secuencias de prueba...")
    
    all_predictions = []
    
    # Procesar en lotes
    total_samples = len(test_df)
    start_time = time.time()
    
    for i in range(0, total_samples, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total_samples)
        batch_seqs = test_df['Sequence'].iloc[i:end]
        
        batch_preds = predict_taxonomy_batch(index, vectorizer, train_df, batch_seqs)
        all_predictions.append(batch_preds)
        
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"Procesadas {end}/{total_samples} secuencias...")
            
    pred_df = pd.concat(all_predictions, ignore_index=True)
    
    # Calcular métricas para cada nivel
    levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus']
    
    print("\n" + "="*50)
    print("PRECISIÓN JERÁRQUICA")
    print("="*50)
    
    for level in levels:
        if level in test_df.columns and level in pred_df.columns:
            y_true = test_df[level].reset_index(drop=True)
            y_pred = pred_df[level]
            
            # Filtrar desconocidos si es necesario, o tratarlos como errores
            acc = accuracy_score(y_true, y_pred)
            print(f"{level:10} Precisión: {acc:.4f} ({acc*100:.2f}%)")
            
    # Reporte detallado para Phylum
    print("\n" + "="*50)
    print("REPORTE DETALLADO PARA PHYLUM")
    print("="*50)
    y_true_phylum = test_df['Phylum'].reset_index(drop=True)
    y_pred_phylum = pred_df['Phylum']
    
    # Obtener clases únicas en conjunto de prueba para evitar reportes enormes si faltan muchas clases
    unique_labels = y_true_phylum.unique()
    
    print(classification_report(y_true_phylum, y_pred_phylum, labels=unique_labels, zero_division=0))
    
    # Matriz de Confusión (Top 10 clases)
    print("\n" + "="*50)
    print("MATRIZ DE CONFUSIÓN (Top 10 Phyla)")
    print("="*50)
    top_classes = y_true_phylum.value_counts().head(10).index
    
    # Filtrar datos para clases principales
    mask = y_true_phylum.isin(top_classes)
    y_true_top = y_true_phylum[mask]
    y_pred_top = y_pred_phylum[mask]
    
    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_classes)
    cm_df = pd.DataFrame(cm, index=top_classes, columns=top_classes)
    
    # Imprimir matriz de confusión legible
    print(cm_df)
    
    print(f"\nTiempo total de evaluación: {time.time() - start_time:.2f}s")

def main():
    index, vectorizer, train_df, test_df = load_artifacts()
    if index is None:
        return
        
    evaluate_model(index, vectorizer, train_df, test_df)

if __name__ == "__main__":
    main()
