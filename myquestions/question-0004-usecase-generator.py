import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.pipeline import Pipeline

def generar_datos_hidraulicos(n_muestras=200):
    """Genera datos sintéticos de sensores IoT respetando el orden temporal."""
    rng = np.random.default_rng(42)
    
    data = {
        'presion_psi': rng.uniform(30, 90, n_muestras),
        'flujo_m3s': rng.uniform(0.1, 5.0, n_muestras),
        'vibracion_hz': rng.uniform(10, 500, n_muestras),
        'cloro_ppm': rng.uniform(0.2, 2.0, n_muestras),
        'fallo': rng.choice([0, 1], n_muestras, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # IMPORTANTE: No se aplica shuffle para no romper la cronología
    X = df.drop(columns=['fallo'])
    y = df['fallo']
    
    return X, y

def analizar_resiliencia_hidraulica(X, y):
    """Pipeline completo con Normalizer, VotingClassifier y validación temporal."""
    
    # 1 y 2. Pipeline: Normalización por filas (Norma L2) + Ensamble de Votación
    pipeline = Pipeline([
        ('normalizer', Normalizer(norm='l2')),
        ('modelo', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000, solver='liblinear')),
                ('kn', KNeighborsClassifier(n_neighbors=3))
            ],
            voting='hard'
        ))
    ])
    
    # 3. Validación temporal con 4 particiones
    tscv = TimeSeriesSplit(n_splits=4)
    
    # 4. Métrica Matthews Correlation Coefficient (MCC)
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    # Evaluación del modelo
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=tscv,
        scoring=mcc_scorer
    )
    
    # 5. Resultado: Promedio del MCC
    return scores.mean()

# --- PRUEBA DEL SISTEMA ---
# X_data, y_data = generar_datos_hidraulicos(300)
# mcc_final = analizar_resiliencia_hidraulica(X_data, y_data)
# print(f"Promedio MCC: {mcc_final:.4f}")
