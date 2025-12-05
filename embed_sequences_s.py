import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

# Configuración
CSV_PATH = 'data/final_dataset.csv'
MODEL_NAME = 'zhihan1996/DNABERT-S'
MILVUS_DB_PATH = 'milvus_db/milvus.db'
COLLECTION_NAME = 'dna_sequences_s'
BATCH_SIZE = 16  # Tamaño de lote conservador para evitar OOM
MAX_LEN = 256 

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
print("Inicializando Milvus...")
try:
    if not os.path.exists(os.path.dirname(MILVUS_DB_PATH)):
        os.makedirs(os.path.dirname(MILVUS_DB_PATH))
    
    client = MilvusClient(uri=MILVUS_DB_PATH)
    
    if client.has_collection(COLLECTION_NAME):
        print(f"Colección '{COLLECTION_NAME}' ya existe.")
    else:
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 500}
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=768,
            id_type="string",
            max_length=512,
            metric_type="COSINE",
            auto_id=False,
            index_params=index_params
        )
        print(f"Colección '{COLLECTION_NAME}' creada con índice HNSW.")
except Exception as e:
    print(f"Error inicializando Milvus: {e}")
    exit(1)

# Función para generar embeddings
def get_embeddings(sequences):
    # Limpiar secuencias: eliminar saltos de línea y espacios en blanco
    sequences = [s.replace('\n', '').strip() for s in sequences]
    
    # Tokenizar
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Usar el embedding del token [CLS] (primer token)
    # El ejemplo del usuario usó: cls_output = outputs[0][:, 0, :]
    embeddings = outputs[0][:, 0, :].cpu().numpy()
    return embeddings

# Procesar CSV en fragmentos
print(f"Procesando {CSV_PATH}...")

# Verificar si el archivo existe
if not os.path.exists(CSV_PATH):
    print(f"Archivo no encontrado: {CSV_PATH}")
    exit(1)

# 1. Cargar y Muestrear 100%
print("Muestreando 100% del conjunto de datos...")
chunks = []
# Usar un tamaño de fragmento más grande para lectura para acelerar E/S, muestrearemos inmediatamente
read_chunk_size = 50000 
try:
    for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=read_chunk_size), desc="Leyendo y muestreando"):
        sampled_chunk = chunk.sample(frac=1.0, random_state=42)
        chunks.append(sampled_chunk)
except Exception as e:
    print(f"Error leyendo CSV: {e}")
    exit(1)

df_sample = pd.concat(chunks)
print(f"Muestreadas {len(df_sample)} secuencias.")

# 2. División Entrenamiento/Prueba
print("Realizando División Entrenamiento/Prueba (80/20)...")
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
train_df['split'] = 'train'
test_df['split'] = 'test'

# Solo procesar train
processing_df = train_df
print(f"Procesando solo conjunto de entrenamiento ({len(processing_df)} secuencias). Test set ignorado.")

# 3. Procesar en lotes
total_rows = len(processing_df)
num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Iniciando generación de embeddings para {total_rows} secuencias en {num_batches} lotes...")

for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Lotes de embeddings"):
    batch_df = processing_df.iloc[i : i + BATCH_SIZE]
    
    try:
        # Limpiar secuencias
        sequences = batch_df['Sequence'].astype(str).tolist()
        headers = batch_df['Header'].astype(str).tolist()
        
        # Preparar metadatos
        metadatas = []
        for _, row in batch_df.iterrows():
            # Manejar valores NaN
            meta = {
                'Kingdom': str(row['Kingdom']) if pd.notna(row['Kingdom']) else "Unknown",
                'Phylum': str(row['Phylum']) if pd.notna(row['Phylum']) else "Unknown",
                'Class': str(row['Class']) if pd.notna(row['Class']) else "Unknown",
                'Order': str(row['Order']) if pd.notna(row['Order']) else "Unknown",
                'Family': str(row['Family']) if pd.notna(row['Family']) else "Unknown",
                'Genus': str(row['Genus']) if pd.notna(row['Genus']) else "Unknown",
                'Species': str(row['Species']) if pd.notna(row['Species']) else "Unknown",
                'split': row['split']
            }
            metadatas.append(meta)
            
        # Generar embeddings
        embeddings = get_embeddings(sequences)
        
        # Agregar a Milvus
        data = []
        for j in range(len(sequences)):
            data.append({
                "id": headers[j],
                "vector": embeddings[j].tolist(),
                "sequence": sequences[j],
                **metadatas[j]
            })
            
        client.upsert(
            collection_name=COLLECTION_NAME,
            data=data
        )
        
    except Exception as e:
        print(f"Error procesando lote comenzando en índice {i}: {e}")
        continue

print("¡Hecho! Embeddings almacenados en Milvus.")
