import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Definir la jerarquía
HIERARCHY = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

class TaxonomicNode:
    def __init__(self, level_name, parent_value=None):
        self.level_name = level_name
        self.parent_value = parent_value
        self.model = None
        self.children = {}
        self.classes_ = []

    def predict(self, X_vectorized):
        if self.model is None:
            return None
        return self.model.predict(X_vectorized)

class ConstantModel:
    def __init__(self, value):
        self.value = value
        self.classes_ = [value]
    def predict(self, X):
        return [self.value] * X.shape[0]

def train_node(node, df, X_vectorized, current_level_idx, min_samples=10):
    """
    Función recursiva para entrenar el árbol jerárquico.
    """
    current_level = HIERARCHY[current_level_idx]
    target = df[current_level]
    
    # Si solo hay una clase, no necesitamos un clasificador, pero aún necesitamos descender
    unique_classes = target.unique()
    if len(unique_classes) == 1:
        # Crear un modelo ficticio que siempre prediga esta clase
        node.model = ConstantModel(unique_classes[0])
        node.classes_ = unique_classes
        
        # Recursión inmediata
        if current_level_idx + 1 < len(HIERARCHY):
            next_level_idx = current_level_idx + 1
            class_label = unique_classes[0]
            
            child_node = TaxonomicNode(HIERARCHY[next_level_idx], parent_value=class_label)
            node.children[class_label] = child_node
            
            # Pasar todos los datos al hijo
            train_node(child_node, df, X_vectorized, next_level_idx, min_samples)
        return

    if len(df) < min_samples:
        return

    print(f"Entrenando nodo: Nivel={current_level}, Padre={node.parent_value}, Muestras={len(df)}, Clases={len(unique_classes)}")

    # Entrenar modelo para este nodo
    # Usando DecisionTreeClassifier por velocidad y naturaleza "tipo árbol"
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    # Necesitamos mapear los índices del df actual a las filas en X_vectorized
    # Dado que X_vectorized es una matriz dispersa global, necesitamos cortarla.
    # ¿Asumiendo que df ha conservado sus índices originales o reiniciamos el índice?
    # Mejor pasar el subconjunto de X correspondiente a df.
    
    # Para evitar el seguimiento complejo de índices, asumamos que X_vectorized está alineado con df
    # Esto es complicado durante la recursión si solo pasamos df.
    # Estrategia: Pasar índices.
    
    clf.fit(X_vectorized, target)
    node.model = clf
    node.classes_ = clf.classes_

    # Si no estamos en el fondo, recursión
    if current_level_idx + 1 < len(HIERARCHY):
        next_level_idx = current_level_idx + 1
        
        for class_label in node.classes_:
            # Filtrar datos para este hijo
            mask = (target == class_label)
            child_df = df[mask]
            child_indices = np.where(mask)[0] # Índices relativos al corte actual de X_vectorized
            
            if len(child_df) > min_samples:
                child_node = TaxonomicNode(HIERARCHY[next_level_idx], parent_value=class_label)
                node.children[class_label] = child_node
                
                # Cortar X para el hijo
                X_child = X_vectorized[child_indices]
                
                train_node(child_node, child_df, X_child, next_level_idx, min_samples)

def predict_hierarchy(node, X_vectorized):
    """
    Predice la ruta completa para una sola muestra o lote.
    Por simplicidad, ¿hacemos la predicción por lotes iterativamente?
    La predicción jerárquica es difícil de vectorizar eficientemente para lotes si el árbol está desequilibrado.
    Hagámoslo fila por fila para mayor claridad o usemos un enfoque basado en máscaras.
    """
    # Esta es una parte compleja. Para una "cosa tipo árbol de decisión", ¿quizás solo entrenarlo sea suficiente por ahora?
    # O implementar un predictor simple de una sola muestra.
    pass

def predict_single(node, x_vec):
    result = {}
    curr = node
    while curr and curr.model:
        pred = curr.model.predict(x_vec)[0]
        result[curr.level_name] = pred
        if pred in curr.children:
            curr = curr.children[pred]
        else:
            break
    return result

def main():
    DATASET_PERCENTAGE = 0.1
    
    print("Cargando datos...")
    try:
        df = pd.read_csv('data/final_dataset.csv')
        if DATASET_PERCENTAGE < 1.0:
            df = df.sample(frac=DATASET_PERCENTAGE, random_state=42)
    except FileNotFoundError:
        print("Datos no encontrados.")
        return

    # Limpiar datos
    df = df.dropna(subset=['Sequence'] + HIERARCHY)
    print(f"Datos cargados: {len(df)} muestras")

    # Vectorizar
    print("Vectorizando...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5), lowercase=False, max_features=5000)
    X = vectorizer.fit_transform(df['Sequence'])
    
    # Dividir entrenamiento/prueba
    # Para jerárquico, es más fácil dividir primero, luego construir el árbol en entrenamiento
    indices = np.arange(len(df))
    X_train, X_test, y_train_idx, y_test_idx = train_test_split(X, indices, test_size=0.2, random_state=42)
    
    df_train = df.iloc[y_train_idx]
    df_test = df.iloc[y_test_idx] # No estrictamente necesario para entrenamiento, pero sí para evaluación

    print("Construyendo Árbol Taxonómico...")
    root = TaxonomicNode(HIERARCHY[0], parent_value="ROOT")
    
    # Entrenar
    train_node(root, df_train, X_train, 0, min_samples=20)
    
    print("Árbol construido.")
    
    # Guardar el modelo
    print("Guardando modelo en taxonomic_tree.pkl...")
    joblib.dump({'root': root, 'vectorizer': vectorizer}, 'taxonomic_tree.pkl')
    print("Modelo guardado.")
    
    # Evaluar en algunas muestras de prueba
    print("\nEvaluando en 5 muestras de prueba:")
    for i in range(5):
        x_sample = X_test[i]
        prediction = predict_single(root, x_sample)
        actual = df_test.iloc[i][HIERARCHY].to_dict()
        
        print(f"Muestra {i}:")
        print(f"  Predicho: {prediction}")
        print(f"  Real:     {actual}")
        print("-" * 30)

if __name__ == "__main__":
    main()
