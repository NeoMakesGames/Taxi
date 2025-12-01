import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import CountVectorizer
import os
import itertools
import sys

def main():
    DATA_PATH = 'data/final_dataset.csv'
    INDEX_FILE = 'sequence_index.ivfpq'
    K_MER_LENGTH = 5
    
    if not os.path.exists(INDEX_FILE):
        print(f"Error: Archivo de índice '{INDEX_FILE}' no encontrado. Ejecute indexing.py primero.")
        return
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Archivo de datos '{DATA_PATH}' no encontrado.")
        return
        
    print(f"Cargando índice desde {INDEX_FILE}...")
    try:
        index = faiss.read_index(INDEX_FILE)
    except Exception as e:
        print(f"Error al cargar el índice: {e}")
        return
        
    print(f"Índice cargado. Vectores totales: {index.ntotal}")
    
    # Configurar vectorizador para coincidir con la configuración de indexación
    print("Configurando vectorizador...")
    bases = ['A', 'C', 'G', 'T']
    vocabulary = [''.join(p) for p in itertools.product(bases, repeat=K_MER_LENGTH)]
    
    vectorizer = CountVectorizer(
        analyzer='char', 
        ngram_range=(K_MER_LENGTH, K_MER_LENGTH), 
        lowercase=False,
        vocabulary=vocabulary
    )
    
    # Obtener una secuencia de consulta
    # Elegir un índice aleatorio
    import random
    QUERY_INDEX = random.randint(0, index.ntotal - 1)
    
    if len(sys.argv) > 1:
        try:
            QUERY_INDEX = int(sys.argv[1])
        except ValueError:
            print("Uso: python validate_index.py [indice_consulta]")
            pass

    print(f"Cargando secuencia de consulta en el índice {QUERY_INDEX}...")
    
    # Leer el encabezado para obtener nombres de columnas
    df_header = pd.read_csv(DATA_PATH, nrows=0)
    columns = df_header.columns
    
    # Leer la fila de consulta específica
    # skiprows=QUERY_INDEX+1 omite el encabezado y las filas anteriores
    try:
        df_query = pd.read_csv(DATA_PATH, skiprows=QUERY_INDEX+1, nrows=1, header=None)
        df_query.columns = columns
    except Exception as e:
        print(f"Error leyendo fila de consulta: {e}")
        return

    if df_query.empty:
        print(f"No se encontraron datos en el índice {QUERY_INDEX}")
        return

    query_sequence = df_query.iloc[0]['Sequence']
    query_species = df_query.iloc[0]['Species']
    
    print(f"Especie de consulta: {query_species}")
    print(f"Secuencia de consulta (primeros 50 caracteres): {query_sequence[:50]}...")
    
    # Vectorizar
    print("Vectorizando consulta...")
    X_query_sparse = vectorizer.transform([query_sequence])
    X_query = X_query_sparse.toarray().astype('float32')
    
    # Buscar
    k = 5
    print(f"Buscando los {k} vecinos más cercanos...")
    D, I = index.search(X_query, k)
    
    print("\nResultados:")
    for rank, idx in enumerate(I[0]):
        distance = D[0][rank]
        if idx == -1:
            continue
            
        print(f"Rango {rank+1}: Índice {idx}, Distancia {distance:.4f}")
        
        # Obtener detalles del vecino desde CSV
        try:
            # Leer la fila específica del vecino
            # skiprows=idx+1 omite encabezado y filas anteriores
            df_neighbor = pd.read_csv(DATA_PATH, skiprows=idx+1, nrows=1, header=None)
            df_neighbor.columns = columns
            
            neighbor_species = df_neighbor.iloc[0]['Species']
            print(f"  Especie: {neighbor_species}")
            
        except Exception as e:
            print(f"  No se pudieron recuperar detalles para el índice {idx}: {e}")

if __name__ == "__main__":
    main()
