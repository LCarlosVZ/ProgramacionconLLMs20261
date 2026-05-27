import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# 1. FUNCIÓN GENERADORA (GROUND TRUTH COMPATIBLE) - EN PRIMERA POSICIÓN
# ---------------------------------------------------------
def generar_caso_de_uso_fraude():
    """
    Genera input y output esperado de forma determinista.
    Garantiza retornar la estructura exacta (dict, float).
    Ubicada en primera posición para evitar errores en el namespace del validador.
    """
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
    
    # Calcular el output exacto usando la función principal corregida
    output_data = detectar_clics_fraudulentos(df, target_col, n_features)
    
    return input_data, output_data


# ---------------------------------------------------------
# 2. GENERADOR DE DATOS ALEATORIOS
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
# 3. FUNCIÓN PRINCIPAL (CÓDIGO COMPAÑERO CORREGIDO)
# ---------------------------------------------------------
def detectar_clics_fraudulentos(df, target_col, n_features):
    """
    Selecciona características mediante RFE y evalúa con GaussianNB.
    Mantiene la compatibilidad con nombres de columnas tras la imputación.
    """
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame")
        
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Validaciones obligatorias de dimensiones y clases
    if n_features > X.shape[1]:
        raise ValueError("n_features no puede ser mayor al número de variables")
    if n_features < 1:
        raise ValueError("n_features debe ser al menos 1")
    if y.nunique() < 2:
        raise ValueError("El target debe tener al menos 2 clases")
    
    # 2. Imputar valores faltantes manteniendo la estructura de DataFrame de Pandas
    imputer = SimpleImputer(strategy='mean')
    X_imputed_array = imputer.fit_transform(X)
    
    # RECONSTRUCCIÓN CRÍTICA: Convertir el array de vuelta a DataFrame con sus columnas originales
    X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)
    
    # 3. Crear pipeline
    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('selector', RFE(
            estimator=LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
            n_features_to_select=n_features
        )),
        ('modelo', GaussianNB())
    ])
    
    # 4. Validación cruzada repetida estratificada
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    # 5. Calcular balanced_accuracy
    scores = cross_val_score(
        pipeline, 
        X_imputed, 
        y,
        cv=rkf,
        scoring='balanced_accuracy'
    )
    
    # 6. Retornar el promedio explícito como float ordinario
    return float(scores.mean())


# ---------------------------------------------------------
# 4. COMPROBACIÓN LOCAL
# ---------------------------------------------------------
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_fraude()
    
    print("=== CONFIGURACIÓN DE PRUEBA ===")
    print(f"Filas: {entrada['df'].shape[0]} | features a seleccionar: {entrada['n_features']}")
    
    resultado = detectar_clics_fraudulentos(
        entrada['df'],
        entrada['target_col'],
        entrada['n_features']
    )
    
    print(f"\nOutput Esperado: {salida_esperada}")
    print(f"Resultado Obtenido: {resultado}")
    
    if np.isclose(resultado, salida_esperada, rtol=1e-5, atol=1e-8):
        print("\n✅ TODO SINCRONIZADO: El error fue resuelto y las estructuras coinciden.")
    else:
        print("\n❌ ALERTA: Sigue existiendo una discrepancia de precisión.")
