import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline

def generar_datos_fraude(n_muestras=200):
    """Genera un escenario aleatorio de clics con variables dispersas."""
    rng = np.random.default_rng(42)
    
    data = {
        'ip_frecuencia': rng.poisson(1, n_muestras),
        'dispositivo_id': rng.integers(0, 50, n_muestras),
        'tipo_red': rng.integers(0, 5, n_muestras),
        'hora_click': rng.uniform(0, 24, n_muestras),
        'tiempo_en_pagina': rng.exponential(10, n_muestras),
        'es_fraude': rng.choice([0, 1], n_muestras, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

def detectar_clics_fraudulentos(df, target_col, n_features):
    """
    Pipeline de Scikit-Learn:
    1. MaxAbsScaler (Escalado disperso)
    2. RFE (Selección con Regresión Logística)
    3. GaussianNB (Modelo final)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Construcción del Pipeline para evitar leakage
    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('selector', RFE(estimator=LogisticRegression(max_iter=1000), 
                         n_features_to_select=n_features)),
        ('modelo', GaussianNB())
    ])
    
    # Validación con 5 splits y 2 repeticiones
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    # Evaluación con balanced_accuracy
    scores = cross_val_score(
        pipeline, 
        X, 
        y, 
        cv=rkf, 
        scoring='balanced_accuracy'
    )
    
    return scores.mean()

# --- EJECUCIÓN ---
# df_fraude = generar_datos_fraude(300)
# resultado = detectar_clics_fraudulentos(df_fraude, 'es_fraude', n_features=3)
# print(f"Balanced Accuracy Promedio: {resultado:.4f}")
