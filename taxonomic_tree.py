import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Define the hierarchy
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
    
    # If only one class, we don't need a classifier, but we still need to traverse down
    unique_classes = target.unique()
    if len(unique_classes) == 1:
        # Create a dummy model that always predicts this class
        node.model = ConstantModel(unique_classes[0])
        node.classes_ = unique_classes
        
        # Recurse immediately
        if current_level_idx + 1 < len(HIERARCHY):
            next_level_idx = current_level_idx + 1
            class_label = unique_classes[0]
            
            child_node = TaxonomicNode(HIERARCHY[next_level_idx], parent_value=class_label)
            node.children[class_label] = child_node
            
            # Pass all data to child
            train_node(child_node, df, X_vectorized, next_level_idx, min_samples)
        return

    if len(df) < min_samples:
        return

    print(f"Entrenando nodo: Nivel={current_level}, Padre={node.parent_value}, Muestras={len(df)}, Clases={len(unique_classes)}")

    # Train model for this node
    # Using DecisionTreeClassifier for speed and "tree-like" nature
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    # We need to map the indices of the current df to the rows in X_vectorized
    # Since X_vectorized is a global sparse matrix, we need to slice it.
    # Assuming df has preserved its original indices or we reset index?
    # Better to pass the subset of X corresponding to df.
    
    # To avoid complex index tracking, let's assume X_vectorized is aligned with df
    # This is tricky during recursion if we just pass df.
    # Strategy: Pass indices.
    
    clf.fit(X_vectorized, target)
    node.model = clf
    node.classes_ = clf.classes_

    # If we are not at the bottom, recurse
    if current_level_idx + 1 < len(HIERARCHY):
        next_level_idx = current_level_idx + 1
        
        for class_label in node.classes_:
            # Filter data for this child
            mask = (target == class_label)
            child_df = df[mask]
            child_indices = np.where(mask)[0] # Indices relative to the current X_vectorized slice
            
            if len(child_df) > min_samples:
                child_node = TaxonomicNode(HIERARCHY[next_level_idx], parent_value=class_label)
                node.children[class_label] = child_node
                
                # Slice X for the child
                X_child = X_vectorized[child_indices]
                
                train_node(child_node, child_df, X_child, next_level_idx, min_samples)

def predict_hierarchy(node, X_vectorized):
    """
    Predice la ruta completa para una sola muestra o lote.
    Por simplicidad, ¿hacemos la predicción por lotes iterativamente?
    La predicción jerárquica es difícil de vectorizar eficientemente para lotes si el árbol está desequilibrado.
    Hagámoslo fila por fila para mayor claridad o usemos un enfoque basado en máscaras.
    """
    # This is a complex part. For a "decision tree like thing", maybe just training it is enough for now?
    # Or implementing a simple single-sample predictor.
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

    # Clean data
    df = df.dropna(subset=['Sequence'] + HIERARCHY)
    print(f"Datos cargados: {len(df)} muestras")

    # Vectorize
    print("Vectorizando...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5), lowercase=False, max_features=5000)
    X = vectorizer.fit_transform(df['Sequence'])
    
    # Split train/test
    # For hierarchical, it's easier to split first, then build tree on train
    indices = np.arange(len(df))
    X_train, X_test, y_train_idx, y_test_idx = train_test_split(X, indices, test_size=0.2, random_state=42)
    
    df_train = df.iloc[y_train_idx]
    df_test = df.iloc[y_test_idx] # Not strictly needed for training, but for eval

    print("Construyendo Árbol Taxonómico...")
    root = TaxonomicNode(HIERARCHY[0], parent_value="ROOT")
    
    # Train
    train_node(root, df_train, X_train, 0, min_samples=20)
    
    print("Árbol construido.")
    
    # Save the model
    print("Guardando modelo en taxonomic_tree.pkl...")
    joblib.dump({'root': root, 'vectorizer': vectorizer}, 'taxonomic_tree.pkl')
    print("Modelo guardado.")
    
    # Evaluate on a few test samples
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
