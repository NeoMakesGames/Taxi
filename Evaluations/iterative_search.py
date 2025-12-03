import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Configuración
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'quietflamingo/dnabert2-no-flashattention'
CHROMA_DB_PATH = 'chroma_db'
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

# Inicializar ChromaDB
print("Conectando a ChromaDB...")
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Conectado a la colección '{COLLECTION_NAME}' con {collection.count()} documentos.")
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
def iterative_search(sequence, collection, k=K_NEIGHBORS):
    # Generar embedding
    embedding = get_embeddings([sequence])[0].tolist()
    
    current_filter = {"split": "train"}
    predicted_taxonomy = {}
    
    print(f"\n--- Iniciando Búsqueda Iterativa ---")
    
    for level in HIERARCHY:
        # Construir cláusula where de ChromaDB
        if len(current_filter) > 1:
            where_clause = {"$and": [{k: v} for k, v in current_filter.items()]}
        else:
            where_clause = current_filter

        # Consultar ChromaDB con filtro actual
        results = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where_clause
        )
        
        if not results['metadatas'] or not results['metadatas'][0]:
            print(f"No se encontraron resultados en el nivel {level}. Deteniendo.")
            break
            
        metadatas = results['metadatas'][0]
        
        # Extraer valores para el nivel actual
        values = [m.get(level, "Unknown") for m in metadatas]
        
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
    # ¿Promedio ponderado donde los niveles superiores (Reino) tienen más peso?
    # ¿O solo un producto de confianzas?
    # El usuario pidió "teniendo en cuenta la importancia decreciente de cada paso en la jerarquía"
    # ¿Esto implica que Reino es MÁS importante que Especie, o viceversa?
    # Usualmente, obtener el Reino correcto es "más fácil" y menos específico. Obtener la Especie correcta es difícil.
    # Si queremos un puntaje único, ¿quizás una suma ponderada de confianzas?
    # Implementemos un puntaje ponderado simple donde Reino tiene peso 1, Filo 1, etc.
    # ¿O quizás pesos decrecientes? "importancia decreciente de cada paso" -> ¿Reino es más importante?
    # Asumamos que Reino (índice 0) es más importante, Especie (índice 6) es menos importante para la clasificación "base",
    # PERO usualmente en taxonomía, la identificación específica es el objetivo.
    # Sin embargo, el prompt dice "importancia decreciente de cada paso".
    # Interpretemos "importancia decreciente" como: Reino (peso alto) -> Especie (peso bajo).
    # Pesos: Reino=7, Filo=6, ..., Especie=1.
    
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
    
    prediction, overall_score = iterative_search(str(row['Sequence']), collection)
    
    print(f"Predicción Final: {prediction}")
    print(f"Confianza Ponderada General: {overall_score:.4f}")
    
    # Verificar corrección
    correct_levels = 0
    for level, data in prediction.items():
        if str(true_tax.get(level)) == data['value']:
            correct_levels += 1
    print(f"Niveles Correctos: {correct_levels}/{len(prediction)} (de predichos)")

