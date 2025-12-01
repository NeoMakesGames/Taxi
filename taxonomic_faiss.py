import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import faiss
import time
import pickle
import os

# Configuración
K_MER_SIZE = 6
DATA_PATH = 'data/final_dataset.csv'
SAMPLE_FRAC = 0.05  # Usar 5% de los datos para demostración y evitar problemas de memoria
SEED = 42
MODEL_DIR = 'faiss_models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_and_preprocess_data(path, frac=1.0):
    print(f"Cargando datos desde {path}...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Archivo {path} no encontrado.")
        return None

    if frac < 1.0:
        print(f"Muestreando {frac*100}% de los datos...")
        df = df.sample(frac=frac, random_state=SEED)
    
    # Eliminar filas con valores faltantes en Sequence o Phylum (como ejemplo de objetivo principal)
    df = df.dropna(subset=['Sequence', 'Phylum'])
    
    print(f"Datos cargados: {len(df)} secuencias.")
    return df

def generate_kmers_vectorizer(sequences, k):
    print(f"Generando {k}-mers...")
    # Analyzer='char' con ngram_range=(k, k) crea k-mers a partir de la cadena
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    X = vectorizer.fit_transform(sequences)
    return X, vectorizer

def build_faiss_index(vectors, batch_size=5000):
    print("Construyendo índice Faiss...")
    d = vectors.shape[1]  # Dimensión de los vectores
    
    # Usando distancia L2 (Euclidiana) para similitud
    index = faiss.IndexFlatL2(d)
    
    # Procesar en lotes para evitar MemoryError
    num_vectors = vectors.shape[0]
    print(f"Agregando {num_vectors} vectores al índice en lotes de {batch_size}...")
    
    for i in range(0, num_vectors, batch_size):
        end = min(i + batch_size, num_vectors)
        # Convertir lote a float32 denso
        batch_dense = vectors[i:end].toarray().astype('float32')
        index.add(batch_dense)
        if (i // batch_size) % 10 == 0:
            print(f"Procesados {end}/{num_vectors} vectores...")
            
    print(f"Índice construido con {index.ntotal} vectores.")
    return index

def predict_taxonomy(index, vectorizer, train_df, query_sequences, k_neighbors=5):
    print(f"Prediciendo taxonomía para {len(query_sequences)} secuencias...")
    
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

def main():
    start_time = time.time()
    
    # 1. Cargar Datos
    df = load_and_preprocess_data(DATA_PATH, SAMPLE_FRAC)
    if df is None:
        return

    # Dividir en entrenamiento (base de datos) y prueba (consulta)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    print(f"Conjunto de entrenamiento: {len(train_df)}, Conjunto de prueba: {len(test_df)}")

    # 2. Generar K-mers y Vectorizar
    # Ajustamos el vectorizador en las secuencias de entrenamiento
    X_train, vectorizer = generate_kmers_vectorizer(train_df['Sequence'], K_MER_SIZE)
    
    # 3. Construir Índice Faiss
    index = build_faiss_index(X_train)
    
    # 4. Guardar Modelos y Datos
    print("\n--- Guardando Modelos y Datos ---")
    
    # Guardar índice Faiss
    index_path = os.path.join(MODEL_DIR, 'taxonomy.index')
    faiss.write_index(index, index_path)
    print(f"Índice Faiss guardado en {index_path}")
    
    # Guardar Vectorizador
    vec_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizador guardado en {vec_path}")
    
    # Guardar Datos de Entrenamiento (Base de Datos de Referencia)
    train_data_path = os.path.join(MODEL_DIR, 'train_data.pkl')
    train_df.to_pickle(train_data_path)
    print(f"Datos de entrenamiento guardados en {train_data_path}")
    
    # Guardar Datos de Prueba (para script de validación)
    test_data_path = os.path.join(MODEL_DIR, 'test_data.pkl')
    test_df.to_pickle(test_data_path)
    print(f"Datos de prueba guardados en {test_data_path}")

    print(f"\nTiempo total de ejecución: {time.time() - start_time:.2f}s")
    print("Ejecute 'python faiss_validation.py' para evaluar el modelo.")

if __name__ == "__main__":
    main()
