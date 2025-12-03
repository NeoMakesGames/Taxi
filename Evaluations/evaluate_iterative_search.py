import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# Configuración
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'zhihan1996/DNABERT-2-117M'
CHROMA_DB_PATH = 'chroma_db'
COLLECTION_NAME = 'dna_sequences'
MAX_LEN = 256 
K_NEIGHBORS = 10 
HIERARCHY = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

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
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Conectado a la colección '{COLLECTION_NAME}' con {collection.count()} documentos.")
except Exception as e:
    print(f"Error conectando a ChromaDB: {e}")
    exit(1)

# Recargar Datos de Prueba
TEST_CACHE_PATH = 'data/test_dataset_cache.csv'

use_cache = False
if os.path.exists(TEST_CACHE_PATH):
    if os.path.exists(CSV_PATH):
        csv_mtime = os.path.getmtime(CSV_PATH)
        cache_mtime = os.path.getmtime(TEST_CACHE_PATH)
        if cache_mtime > csv_mtime:
            use_cache = True
            print(f"Caché es válido (más reciente que CSV).")
        else:
            print("Caché desactualizado (CSV es más reciente). Recargando...")
    else:
        use_cache = True

if use_cache:
    print(f"Cargando conjunto de prueba desde caché: {TEST_CACHE_PATH}")
    test_df = pd.read_csv(TEST_CACHE_PATH)
    print(f"Tamaño del conjunto de prueba cargado: {len(test_df)}")
else:
    print(f"Recargando conjunto de datos para reconstruir la división de prueba...")
    if not os.path.exists(CSV_PATH):
        print(f"Archivo no encontrado: {CSV_PATH}")
        exit(1)

    chunks = []
    read_chunk_size = 50000  # Igualar a embed_sequences.py para consistencia
    cols_to_use = ['Sequence'] + HIERARCHY
    
    try:
        for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size, usecols=cols_to_use), desc="Leyendo y muestreando"):
            sampled_chunk = chunk.sample(frac=1.0, random_state=42) # Igualar a embed_sequences.py (100%)
            chunks.append(sampled_chunk)
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        exit(1)

    df_sample = pd.concat(chunks)
    train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
    print(f"Tamaño total del conjunto de prueba generado: {len(test_df)}")
    
    # Guardar en caché para futuras ejecuciones
    test_df.to_csv(TEST_CACHE_PATH, index=False)
    print(f"Conjunto de prueba guardado en caché: {TEST_CACHE_PATH}")

# Limitar a 30 muestras para prueba
test_df = test_df.head(30)
print(f"Limitando evaluación a {len(test_df)} muestras.")

# Función para generar embeddings en lotes

def get_embeddings_batched(sequences, batch_size=8):
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Generando Embeddings"):
        batch = sequences[i:i+batch_size]
        batch = [s.replace('\n', '').strip() for s in batch]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs[0][:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Lógica de Búsqueda Iterativa (Optimizada)
def iterative_search_from_embedding(embedding, collection, k=K_NEIGHBORS):
    print("DEBUG: Starting iterative search for a sample")
    current_filter = {} # Removed "split": "train" as DB only contains train data
    predicted_taxonomy = {}
    
    for level in HIERARCHY:
        # Construir cláusula where de ChromaDB
        where_clause = None
        if current_filter:
            if len(current_filter) > 1:
                where_clause = {"$and": [{k: v} for k, v in current_filter.items()]}
            else:
                where_clause = current_filter

        # Consultar ChromaDB con filtro actual
        try:
            print(f"DEBUG: Querying level {level} with filter {where_clause}")
            results = collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=k,
                where=where_clause,
                include=['metadatas'] # Optimización: Solo obtener metadatos
            )
            print(f"DEBUG: Query returned for {level}")
        except Exception as e:
            print(f"Error querying ChromaDB at level {level}: {e}")
            break
        
        if not results['metadatas'] or not results['metadatas'][0]:
            break
            
        metadatas = results['metadatas'][0]
        values = [m.get(level, "Unknown") for m in metadatas]
        
        if not values:
            break

        counts = Counter(values)
        most_common_val, count = counts.most_common(1)[0]
        confidence = count / len(values)
        
        if most_common_val != "Unknown":
            current_filter[level] = most_common_val
            predicted_taxonomy[level] = {
                "value": most_common_val,
                "confidence": confidence
            }
        else:
            break
            
    # Calcular puntaje de confianza general
    total_weight = 0
    weighted_confidence_sum = 0
    weights = {'Kingdom': 7, 'Phylum': 6, 'Class': 5, 'Order': 4, 'Family': 3, 'Genus': 2, 'Species': 1}
    
    for level, data in predicted_taxonomy.items():
        w = weights.get(level, 1)
        weighted_confidence_sum += data['confidence'] * w
        total_weight += w
        
    overall_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0
    
    return predicted_taxonomy, overall_confidence

# Bucle de Evaluación
print(f"Iniciando evaluación en {len(test_df)} muestras...")

# Pre-calcular embeddings
print("Pre-calculando embeddings para el conjunto de prueba...")
test_sequences = test_df['Sequence'].astype(str).tolist()
test_embeddings = get_embeddings_batched(test_sequences, batch_size=8)

if np.isnan(test_embeddings).any():
    print("ERROR: Embeddings contain NaNs!")
    exit(1)
if np.isinf(test_embeddings).any():
    print("ERROR: Embeddings contain Infs!")
    exit(1)
print(f"Embeddings shape: {test_embeddings.shape}")

y_true = {level: [] for level in HIERARCHY}
y_pred = {level: [] for level in HIERARCHY}
confidences = []

def process_single_sample(args):
    embedding, (_, row) = args
    true_tax = {level: str(row[level]) if pd.notna(row[level]) else "Unknown" for level in HIERARCHY}
    
    prediction, overall_score = iterative_search_from_embedding(embedding, collection)
    
    pred_result = {}
    for level in HIERARCHY:
        if level in prediction:
            pred_result[level] = prediction[level]['value']
        else:
            pred_result[level] = "Unpredicted"
            
    return true_tax, pred_result, overall_score

print("Ejecutando búsqueda iterativa secuencialmente...")
# Preparar argumentos
args_list = list(zip(test_embeddings, test_df.iterrows()))

for i, args in enumerate(tqdm(args_list, desc="Evaluando")):
    try:
        true_tax, pred_result, overall_score = process_single_sample(args)
        confidences.append(overall_score)
        
        print(f"Sample {i+1}/{len(args_list)} processed. Confidence: {overall_score:.4f}")
        
        for level in HIERARCHY:
            y_true[level].append(true_tax[level])
            y_pred[level].append(pred_result[level])
    except Exception as e:
        print(f"Error procesando muestra: {e}")


# Calcular Métricas
print("\n--- Resultados de Evaluación ---")
metrics = []

for level in HIERARCHY:
    # ¿Filtrar 'Unknown' en la verdad terreno si se desea?
    # Usualmente queremos evaluar contra la verdad terreno conocida.
    # Pero aquí mantenemos todo.
    
    acc = accuracy_score(y_true[level], y_pred[level])
    
    # Promedio ponderado para multi-clase
    prec = precision_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    rec = recall_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    f1 = f1_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    
    metrics.append({
        "Level": level,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

df_metrics = pd.DataFrame(metrics)
print(df_metrics.to_string(index=False))

print(f"\nConfianza General Promedio: {np.mean(confidences):.4f}")

# Guardar resultados en CSV
df_metrics.to_csv("iterative_search_results.csv", index=False)
print("Resultados guardados en iterative_search_results.csv")
