import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import CountVectorizer
import time
import os

def main():
    # Configuración
    DATA_PATH = 'data/final_dataset.csv'
    INDEX_FILE = 'sequence_index.ivfpq'
    K_MER_LENGTH = 5
    N_LIST = 100  # Número de clústeres (celdas de Voronoi)
    M = 32        # Número de sub-cuantizadores (debe dividir la dimensión)
    N_BITS = 8    # Bits por sub-cuantizador
    CHUNK_SIZE = 10000 # Procesar 10k filas a la vez para ahorrar memoria

    print("Cargando configuración de datos...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} no encontrado.")
        return

    # Configuración de preprocesamiento
    print("Configurando vectorizador...")
    
    # Generar todos los posibles 5-mers para asegurar dimensión fija
    import itertools
    bases = ['A', 'C', 'G', 'T']
    vocabulary = [''.join(p) for p in itertools.product(bases, repeat=K_MER_LENGTH)]
    
    # Usando CountVectorizer con vocabulario fijo
    vectorizer = CountVectorizer(
        analyzer='char', 
        ngram_range=(K_MER_LENGTH, K_MER_LENGTH), 
        lowercase=False,
        vocabulary=vocabulary
    )

    # Inicializar variables
    index = None
    total_processed = 0
    start_time_total = time.time()

    # Procesar en fragmentos
    print(f"Procesando datos en fragmentos de {CHUNK_SIZE}...")
    
    # Crear iterador
    chunk_iterator = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE)
    
    for i, df_chunk in enumerate(chunk_iterator):
        chunk_start = time.time()
        print(f"Procesando fragmento {i+1} ({len(df_chunk)} filas)...")
        
        # Vectorizar fragmento
        X_sparse = vectorizer.transform(df_chunk['Sequence'])
        X = X_sparse.toarray().astype('float32')
        
        # Inicializar y entrenar índice en el primer fragmento
        if index is None:
            d = X.shape[1]
            print(f"Dimensión de características: {d}")
            
            if d % M != 0:
                print(f"Error: La dimensión {d} no es divisible por M={M}. Ajuste M.")
                return

            print("Construyendo y entrenando índice IVFPQ...")
            quantizer = faiss.IndexFlatL2(d) 
            index = faiss.IndexIVFPQ(quantizer, d, N_LIST, M, N_BITS)
            
            train_start = time.time()
            index.train(X)
            print(f"El entrenamiento tomó {time.time() - train_start:.2f}s")
        
        # Agregar vectores al índice
        index.add(X)
        total_processed += len(df_chunk)
        print(f"Fragmento {i+1} procesado en {time.time() - chunk_start:.2f}s. Vectores totales: {index.ntotal}")

    # Guardar el índice
    print(f"Guardando índice en {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    
    print(f"Tiempo total de procesamiento: {time.time() - start_time_total:.2f}s")
    print(f"El índice contiene {index.ntotal} vectores.")

if __name__ == "__main__":
    main()
