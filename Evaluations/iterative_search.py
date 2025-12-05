import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Configuración
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'quietflamingo/dnabert2-no-flashattention'
MILVUS_DB_PATH = 'milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences'
MAX_LEN = 256 
K_NEIGHBORS = 10 # El usuario mencionó "top 10"
TEST_SAMPLE_SIZE = 10
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

# Función para generar embeddings
def get_embeddings(sequences):
    sequences = [s.replace('\n', '').strip() for s in sequences]
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs[0][:, 0, :].cpu().numpy()
    return embeddings

# Recargar Datos de Prueba (Misma lógica que test_search.py)
print(f"Recargando conjunto de datos para reconstruir la división de prueba...")
if not os.path.exists(CSV_PATH):
    print(f"Archivo no encontrado: {CSV_PATH}")
    exit(1)

chunks = []
read_chunk_size = 50000 
try:
    for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size), desc="Leyendo y muestreando"):
        sampled_chunk = chunk.sample(frac=0.1, random_state=42)
        chunks.append(sampled_chunk)
except Exception as e:
    print(f"Error leyendo CSV: {e}")
    exit(1)

df_sample = pd.concat(chunks)
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
print(f"Tamaño total del conjunto de prueba: {len(test_df)}")

# Lógica de Búsqueda Iterativa
def iterative_search(sequence, client, k=K_NEIGHBORS):
    # Generar embedding
    embedding = get_embeddings([sequence])[0].tolist()
    
    current_filter = {}
    predicted_taxonomy = {}
    
    print(f"\n--- Iniciando Búsqueda Iterativa ---")
    
    # Dynamic K: Start at K_NEIGHBORS and decay to 1
    k_values = np.linspace(k, 1, len(HIERARCHY))
    k_values = [int(round(x)) for x in k_values]
    k_values = [max(1, x) for x in k_values]

    for i, level in enumerate(HIERARCHY):
        current_k = k_values[i]

        # Construir filtro de Milvus
        filter_expr = ""
        if current_filter:
            conditions = [f'{key} == "{value}"' for key, value in current_filter.items()]
            filter_expr = " and ".join(conditions)

        # Consultar Milvus con filtro actual
        try:
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[embedding],
                limit=current_k,
                filter=filter_expr,
                output_fields=[level]
            )
        except Exception as e:
            print(f"Error querying Milvus at level {level}: {e}")
            break
        
        if not results or not results[0]:
            print(f"No se encontraron resultados en el nivel {level}. Deteniendo.")
            break
            
        hits = results[0]
        
        # Extraer valores para el nivel actual
        values = [hit['entity'].get(level, "Unknown") for hit in hits]
        
        # Contar ocurrencias
        counts = Counter(values)
        most_common_val, count = counts.most_common(1)[0]
        confidence = count / len(values)
        
        print(f"Nivel: {level}, Valor superior: {most_common_val}, Confianza: {confidence:.2f} ({count}/{len(values)})")
        
        # Actualizar filtro y predicción independientemente de la confianza
        if most_common_val != "Unknown":
            current_filter[level] = most_common_val
            predicted_taxonomy[level] = {
                "value": most_common_val,
                "confidence": confidence
            }
            print(f"-> Bloqueando {level} = {most_common_val} (Confianza: {confidence:.2f})")
        else:
            print(f"-> Valor desconocido. No se puede refinar más.")
            break
            
    # Calcular puntaje de confianza general
    total_weight = 0
    weighted_confidence_sum = 0
    
    weights = {
        'Kingdom': 7, 'Phylum': 6, 'Class': 5, 
        'Order': 4, 'Family': 3, 'Genus': 2, 'Species': 1
    }
    
    for level, data in predicted_taxonomy.items():
        w = weights.get(level, 1)
        weighted_confidence_sum += data['confidence'] * w
        total_weight += w
        
    overall_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0
    
    return predicted_taxonomy, overall_confidence

# Ejecutar en algunas muestras
print(f"Ejecutando búsqueda iterativa en {TEST_SAMPLE_SIZE} muestras...")
test_subset = test_df.head(TEST_SAMPLE_SIZE)

for idx, row in test_subset.iterrows():
    print(f"\nID de Secuencia de Consulta: {row.name}") # Asumiendo que el índice es significativo o solo usar índice de fila
    true_tax = {level: row[level] for level in HIERARCHY}
    print(f"Taxonomía Verdadera: {true_tax}")
    
    prediction, overall_score = iterative_search(str(row['Sequence']), client)
    
    print(f"Predicción Final: {prediction}")
    print(f"Confianza Ponderada General: {overall_score:.4f}")
    
    # Verificar corrección
    correct_levels = 0
    for level, data in prediction.items():
        if str(true_tax.get(level)) == data['value']:
            correct_levels += 1
    print(f"Niveles Correctos: {correct_levels}/{len(prediction)} (de predichos)")

