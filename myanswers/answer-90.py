import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def optimizar_dimensiones(df, target_col, n_componentes=2):
    """
    Realiza limpieza, estandarización, reducción de dimensionalidad con PCA
    y evaluación de importancia de variables con Random Forest.
    """
    # 1. Limpieza inicial: eliminar filas con NaNs para evitar errores
    df_clean = df.dropna().copy()
    
    # Separación en X e y
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # 2. Estandarización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Reducción de ruido con PCA
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Selección de importancia con RandomForest (usando X_scaled original)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_scaled, y)
    feature_importances = rf.feature_importances_
    
    return X_pca, feature_importances
