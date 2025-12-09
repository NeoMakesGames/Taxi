import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# Configuración
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'zhihan1996/DNABERT-S'
MILVUS_DB_PATH = 'milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences_s'
MAX_LEN = 256
K_NEIGHBORS = 1
K_ENTROPY = 10
HIERARCHY = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
EMBEDDINGS_PATH = 'data/test_embeddings_s.npy'

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

# Inicializar Milvus
print("Conectando a Milvus...")
try:
    client = MilvusClient(uri=MILVUS_DB_PATH)
    print(f"Conectado a Milvus.")
except Exception as e:
    print(f"Error conectando a Milvus: {e}")
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
    read_chunk_size = 50000 
    cols_to_use = ['Sequence'] + HIERARCHY
    
    try:
        for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size, usecols=cols_to_use), desc="Leyendo y muestreando"):
            sampled_chunk = chunk.sample(frac=1.0, random_state=42)
            chunks.append(sampled_chunk)
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        exit(1)

    df_sample = pd.concat(chunks)
    train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
    print(f"Tamaño total del conjunto de prueba generado: {len(test_df)}")
    
    test_df.to_csv(TEST_CACHE_PATH, index=False)
    print(f"Conjunto de prueba guardado en caché: {TEST_CACHE_PATH}")

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

# Lógica de Búsqueda Simple (k-NN)
def simple_search_batched(embeddings, client, k=K_NEIGHBORS, batch_size=100):
    all_predictions = []
    all_confidences = []
    all_distances = []
    all_entropies = []
    
    total_samples = len(embeddings)
    k_search = max(k, K_ENTROPY)
    
    for i in tqdm(range(0, total_samples, batch_size), desc="Buscando en Milvus"):
        batch_embeddings = embeddings[i:i+batch_size]
        
        try:
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=batch_embeddings,
                limit=k_search,
                output_fields=HIERARCHY + ["distance"]
            )
        except Exception as e:
            print(f"Error querying Milvus: {e}")
            # Append empty results for this batch to maintain alignment
            for _ in range(len(batch_embeddings)):
                all_predictions.append({level: "Unknown" for level in HIERARCHY})
                all_confidences.append({level: 0.0 for level in HIERARCHY})
                all_distances.append(float('inf'))
                all_entropies.append({level: 0.0 for level in HIERARCHY})
            continue

        for hits in results:
            predicted_taxonomy = {}
            confidences = {}
            entropies = {}
            
            if not hits:
                all_predictions.append({level: "Unknown" for level in HIERARCHY})
                all_confidences.append({level: 0.0 for level in HIERARCHY})
                all_distances.append(float('inf'))
                all_entropies.append({level: 0.0 for level in HIERARCHY})
                continue

            # Distance to nearest neighbor
            nearest_dist = hits[0]['distance']
            all_distances.append(nearest_dist)

            # Prediction & Voting Confidence (using k neighbors)
            hits_k = hits[:k]
            for level in HIERARCHY:
                values = [hit['entity'].get(level, "Unknown") for hit in hits_k]
                if not values:
                    predicted_taxonomy[level] = "Unknown"
                    confidences[level] = 0.0
                    continue
                    
                counts = Counter(values)
                most_common_val, count = counts.most_common(1)[0]
                predicted_taxonomy[level] = most_common_val
                confidences[level] = count / len(values)
            
            # Entropy Confidence (using K_ENTROPY neighbors)
            hits_entropy = hits[:K_ENTROPY]
            for level in HIERARCHY:
                values = [hit['entity'].get(level, "Unknown") for hit in hits_entropy]
                if not values:
                    entropies[level] = 0.0
                    continue
                
                counts = Counter(values)
                probs = [c / len(values) for c in counts.values()]
                entropies[level] = entropy(probs, base=2)

            all_predictions.append(predicted_taxonomy)
            all_confidences.append(confidences)
            all_entropies.append(entropies)
        
    return all_predictions, all_confidences, all_distances, all_entropies

# Bucle de Evaluación
print(f"Iniciando evaluación simple en {len(test_df)} muestras...")

# Pre-calcular embeddings
if os.path.exists(EMBEDDINGS_PATH):
    print(f"Cargando embeddings pre-calculados desde {EMBEDDINGS_PATH}...")
    test_embeddings = np.load(EMBEDDINGS_PATH)
    if len(test_embeddings) != len(test_df):
        print(f"Advertencia: El número de embeddings ({len(test_embeddings)}) no coincide con el número de muestras de prueba ({len(test_df)}). Recalculando...")
        test_sequences = test_df['Sequence'].astype(str).tolist()
        test_embeddings = get_embeddings_batched(test_sequences, batch_size=128)
else:
    print("Pre-calculando embeddings para el conjunto de prueba...")
    test_sequences = test_df['Sequence'].astype(str).tolist()
    test_embeddings = get_embeddings_batched(test_sequences, batch_size=128)

y_true = {level: [] for level in HIERARCHY}
y_pred = {level: [] for level in HIERARCHY}
y_conf = {level: [] for level in HIERARCHY}
y_dist = []
y_entropy = {level: [] for level in HIERARCHY}

print("Ejecutando búsqueda simple en lote...")
batch_predictions, batch_confidences, batch_distances, batch_entropies = simple_search_batched(test_embeddings, client)

for i, ((_, row), prediction, confidences, dist, entropies) in enumerate(zip(test_df.iterrows(), batch_predictions, batch_confidences, batch_distances, batch_entropies)):
    true_tax = {level: str(row[level]) if pd.notna(row[level]) else "Unknown" for level in HIERARCHY}
    
    y_dist.append(dist)
    
    for level in HIERARCHY:
        y_true[level].append(true_tax[level])
        y_pred[level].append(prediction.get(level, "Unpredicted"))
        y_conf[level].append(confidences.get(level, 0.0))
        y_entropy[level].append(entropies.get(level, 0.0))

# Calcular Métricas
print("\n--- Resultados de Evaluación Simple (DNABERT-S) ---")
metrics = []

avg_distance = np.mean(y_dist)

for level in HIERARCHY:
    acc = accuracy_score(y_true[level], y_pred[level])
    prec = precision_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    rec = recall_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    f1 = f1_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    avg_conf = np.mean(y_conf[level])
    avg_entropy = np.mean(y_entropy[level])
    
    metrics.append({
        "Level": level,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Avg Confidence": avg_conf,
        "Avg Entropy": avg_entropy,
        "Avg Distance": avg_distance # Same for all levels as it's based on nearest neighbor
    })

df_metrics = pd.DataFrame(metrics)
print(df_metrics.to_string(index=False))

# Calculate overall confidence
all_confidences = [c for level_confs in y_conf.values() for c in level_confs]
overall_confidence = np.mean(all_confidences)
print(f"\nConfianza General (Promedio de todos los niveles y muestras): {overall_confidence:.4f}")
print(f"Distancia Promedio al Vecino más Cercano: {avg_distance:.4f}")

# Guardar resultados en CSV
df_metrics.to_csv("Evaluations/simple_search_results_s.csv", index=False)
print("Resultados guardados en Evaluations/simple_search_results_s.csv")
