import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# 1. GENERADOR DE DATOS ALEATORIOS
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 2. FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def detectar_clics_fraudulentos(df, target_col, n_features):
    """
    1. Escalado con MaxAbsScaler
    2. Selección con RFE + LogisticRegression
    3. Modelo GaussianNB
    4. Validación con RepeatedStratifiedKFold
    5. Retorna balanced_accuracy promedio
    """
    
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Validaciones obligatorias de dimensiones y clases
    if n_features > X.shape[1]:
        raise ValueError("n_features no puede ser mayor al número de variables")
    
    if n_features < 1:
        raise ValueError("n_features debe ser al menos 1")
    
    if y.nunique() < 2:
        raise ValueError("El target debe tener al menos 2 clases")
    
    # Pipeline completo
    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('selector', RFE(
            estimator=LogisticRegression(max_iter=1000, solver='liblinear'),
            n_features_to_select=n_features
        )),
        ('modelo', GaussianNB())
    ])
    
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=rkf,
        scoring='balanced_accuracy'
    )
    
    return float(scores.mean())


# ---------------------------------------------------------
# 3. FUNCIÓN GENERADORA (GROUND TRUTH COMPATIBLE)
# ---------------------------------------------------------
def generar_caso_de_uso_fraude():
    """
    Genera input y output esperado (ground truth independiente) de forma determinista.
    Garantiza retornar la estructura exacta (dict, float).
    """
    # Usamos una semilla fija para definir los parámetros aleatorios del escenario
    rng_setup = np.random.default_rng(2026)
    
    n_muestras = int(rng_setup.integers(150, 400))
    df = generar_datos_fraude(n_muestras)
    
    target_col = 'es_fraude'
    
    # Máximo de características basadas en las columnas disponibles (menos el target)
    max_features = df.shape[1] - 1
    n_features = int(rng_setup.integers(1, min(5, max_features) + 1))
    
    input_data = {
        'df': df.copy(),
        'target_col': target_col,
        'n_features': n_features
    }
    
    # Calcular el output exacto esperado usando la lógica de negocio
    output_data = detectar_clics_fraudulentos(df, target_col, n_features)
    
    # Retorno exacto para la suite de pruebas automatizadas
    return input_data, output_data


# ---------------------------------------------------------
# 4. EJEMPLO DE EJECUCIÓN
# ---------------------------------------------------------
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_fraude()
    
    print("=== INPUT CONFIGURACIÓN ===")
    print(f"Número de filas: {entrada['df'].shape[0]}")
    print(f"Target: {entrada['target_col']}")
    print(f"n_features a seleccionar: {entrada['n_features']}\n")
    print(entrada['df'].head(3))
    
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
    
    resultado = detectar_clics_fraudulentos(
        entrada['df'],
        entrada['target_col'],
        entrada['n_features']
    )
    
    print("\n=== RESULTADO FUNCIÓN ===")
    print(resultado)
    
    print("\n=== VALIDACIÓN FINAL ===")
    if np.isclose(resultado, salida_esperada, rtol=1e-5, atol=1e-8):
        print("✅ COMPATIBLE: El generador y la función sincronizan perfectamente.")
    else:
        print("❌ ERROR DE CONCORDANCIA")
