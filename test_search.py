import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Configuración
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'quietflamingo/dnabert2-no-flashattention'
CHROMA_DB_PATH = 'chroma_db'
COLLECTION_NAME = 'dna_sequences'
MAX_LEN = 256 
K_NEIGHBORS = 5
TEST_SAMPLE_SIZE = 100 # Número de muestras de prueba para evaluar

# Inicializar Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Inicializar Modelo y Tokenizador
print("Cargando modelo y tokenizador...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    model.eval()
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    exit(1)

# Inicializar ChromaDB
print("Conectando a ChromaDB...")
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Conectado a la colección '{COLLECTION_NAME}' con {count} documentos.")
        if count == 0:
            print("Advertencia: La colección está vacía. Por favor ejecute embed_sequences.py primero.")
            exit(1)
    except Exception as e:
        print(f"Colección '{COLLECTION_NAME}' no encontrada. Por favor ejecute embed_sequences.py primero.")
        exit(1)
except Exception as e:
    print(f"Error conectando a ChromaDB: {e}")
    exit(1)

# Función para generar embeddings
def get_embeddings(sequences):
    sequences = [s.replace('\n', '').strip() for s in sequences]
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs[0][:, 0, :].cpu().numpy()
    return embeddings

# Recargar Datos de Prueba
print(f"Recargando conjunto de datos para reconstruir la división de prueba...")
if not os.path.exists(CSV_PATH):
    print(f"Archivo no encontrado: {CSV_PATH}")
    exit(1)

chunks = []
read_chunk_size = 50000 
try:
    # Debemos usar exactamente la misma lógica que embed_sequences.py para obtener la misma división
    for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size), desc="Leyendo y muestreando"):
        sampled_chunk = chunk.sample(frac=0.1, random_state=42)
        chunks.append(sampled_chunk)
except Exception as e:
    print(f"Error leyendo CSV: {e}")
    exit(1)

df_sample = pd.concat(chunks)
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
print(f"Tamaño total del conjunto de prueba: {len(test_df)}")

# Realizar Búsqueda
print(f"Ejecutando búsqueda en un subconjunto de {TEST_SAMPLE_SIZE} muestras de prueba...")
test_subset = test_df.head(TEST_SAMPLE_SIZE)

results = []

for idx, row in tqdm(test_subset.iterrows(), total=len(test_subset), desc="Buscando"):
    sequence = str(row['Sequence'])
    true_metadata = {
        'Kingdom': row['Kingdom'],
        'Phylum': row['Phylum'],
        'Class': row['Class'],
        'Order': row['Order'],
        'Family': row['Family'],
        'Genus': row['Genus'],
        'Species': row['Species']
    }
    
    # Generar embedding
    try:
        embedding = get_embeddings([sequence])[0]
    except Exception as e:
        print(f"Error generando embedding de secuencia: {e}")
        continue
        
    # Consultar ChromaDB
    # Filtrar por split='train' para asegurar que recuperamos vecinos del conjunto de entrenamiento
    query_result = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=K_NEIGHBORS,
        where={"split": "train"} 
    )
    
    # Analizar resultados
    if not query_result['metadatas'] or not query_result['metadatas'][0]:
        continue
        
    # Obtener la mejor coincidencia (vecino más cercano)
    top_match = query_result['metadatas'][0][0]
    distance = query_result['distances'][0][0]
    
    match_correct = {}
    for level in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']:
        # Manejar NaN/None
        true_val = str(true_metadata[level]) if pd.notna(true_metadata[level]) else "Unknown"
        pred_val = top_match.get(level, "Unknown")
        match_correct[level] = (true_val == pred_val)
    
    results.append(match_correct)

# Calcular Precisión
if results:
    df_results = pd.DataFrame(results)
    print("\n--- Precisión de Búsqueda (Top-1 Vecino del Conjunto de Entrenamiento) ---")
    print(f"Evaluado en {len(results)} muestras.")
    for col in df_results.columns:
        acc = df_results[col].mean()
        print(f"{col}: {acc:.2%}")
else:
    print("No se generaron resultados.")
