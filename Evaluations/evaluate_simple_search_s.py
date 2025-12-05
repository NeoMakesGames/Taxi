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
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# Configuración
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'zhihan1996/DNABERT-S'
MILVUS_DB_PATH = 'milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences_s'
MAX_LEN = 256
K_NEIGHBORS = 1
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

# Limitar a 300 muestras para prueba
test_df = test_df.head(300)
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

# Lógica de Búsqueda Simple (k-NN)
def simple_search_batched(embeddings, client, k=K_NEIGHBORS):
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=embeddings,
            limit=k,
            output_fields=HIERARCHY
        )
    except Exception as e:
        print(f"Error querying Milvus: {e}")
        return [], []
    
    all_predictions = []
    all_confidences = []

    for hits in results:
        predicted_taxonomy = {}
        confidences = {}
        
        if not hits:
            all_predictions.append({level: "Unknown" for level in HIERARCHY})
            all_confidences.append({level: 0.0 for level in HIERARCHY})
            continue

        # Votación mayoritaria para cada nivel
        for level in HIERARCHY:
            values = [hit['entity'].get(level, "Unknown") for hit in hits]
            if not values:
                predicted_taxonomy[level] = "Unknown"
                confidences[level] = 0.0
                continue
                
            counts = Counter(values)
            most_common_val, count = counts.most_common(1)[0]
            predicted_taxonomy[level] = most_common_val
            confidences[level] = count / len(values)
        
        all_predictions.append(predicted_taxonomy)
        all_confidences.append(confidences)
        
    return all_predictions, all_confidences

# Bucle de Evaluación
print(f"Iniciando evaluación simple en {len(test_df)} muestras...")

# Pre-calcular embeddings
print("Pre-calculando embeddings para el conjunto de prueba...")
test_sequences = test_df['Sequence'].astype(str).tolist()
test_embeddings = get_embeddings_batched(test_sequences, batch_size=8)

y_true = {level: [] for level in HIERARCHY}
y_pred = {level: [] for level in HIERARCHY}
y_conf = {level: [] for level in HIERARCHY}

print("Ejecutando búsqueda simple en lote...")
batch_predictions, batch_confidences = simple_search_batched(test_embeddings, client)

for i, ((_, row), prediction, confidences) in enumerate(zip(test_df.iterrows(), batch_predictions, batch_confidences)):
    true_tax = {level: str(row[level]) if pd.notna(row[level]) else "Unknown" for level in HIERARCHY}
    
    for level in HIERARCHY:
        y_true[level].append(true_tax[level])
        y_pred[level].append(prediction.get(level, "Unpredicted"))
        y_conf[level].append(confidences.get(level, 0.0))

# Calcular Métricas
print("\n--- Resultados de Evaluación Simple (DNABERT-S) ---")
metrics = []

for level in HIERARCHY:
    acc = accuracy_score(y_true[level], y_pred[level])
    prec = precision_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    rec = recall_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    f1 = f1_score(y_true[level], y_pred[level], average='weighted', zero_division=0)
    avg_conf = np.mean(y_conf[level])
    
    metrics.append({
        "Level": level,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Avg Confidence": avg_conf
    })

df_metrics = pd.DataFrame(metrics)
print(df_metrics.to_string(index=False))

# Calculate overall confidence
all_confidences = [c for level_confs in y_conf.values() for c in level_confs]
overall_confidence = np.mean(all_confidences)
print(f"\nConfianza General (Promedio de todos los niveles y muestras): {overall_confidence:.4f}")

# Guardar resultados en CSV
df_metrics.to_csv("Evaluations/simple_search_results_s.csv", index=False)
print("Resultados guardados en Evaluations/simple_search_results_s.csv")
