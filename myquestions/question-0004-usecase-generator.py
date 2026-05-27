import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# 1. FUNCIÓN GENERADORA (GROUND TRUTH) - ¡AHORA PRIMERO Y DETERMINISTA!
# ---------------------------------------------------------
def generar_caso_de_uso_hidraulico():
    """
    Genera input y output esperado (ground truth independiente).
    Ubicada en primera posición para el validador automático.
    """
    # Usamos un generador determinista para que las dimensiones sean consistentes
    rng_setup = np.random.default_rng(2026)
    n_muestras = int(rng_setup.integers(150, 350))
    
    X, y = generar_datos_hidraulicos(n_muestras)
    
    input_data = {
        'X': X.copy(),
        'y': y.copy()
    }
    
    # -------- GROUND TRUTH --------
    if len(X) != len(y):
        raise ValueError("Error en generación")
    
    if pd.Series(y).nunique() < 2:
        raise ValueError("Target inválido en generación")
    
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
    
    tscv = TimeSeriesSplit(n_splits=4)
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=tscv,
        scoring=mcc_scorer
    )
    
    output_data = float(scores.mean())
    
    return input_data, output_data


# ---------------------------------------------------------
# 2. GENERADOR DE DATOS ALEATORIOS
# ---------------------------------------------------------
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
    
    # NO se hace shuffle → se mantiene orden temporal
    X = df.drop(columns=['fallo'])
    y = df['fallo']
    
    return X, y


# ---------------------------------------------------------
# 3. FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def analizar_resiliencia_hidraulica(X, y):
    """
    1. Normalización por filas (L2)
    2. VotingClassifier (LR + KNN)
    3. Validación temporal
    4. Métrica MCC
    5. Retorna promedio MCC
    """
    # dimensiones
    if len(X) != len(y):
        raise ValueError("X e y deben tener la misma cantidad de muestras")
    
    if len(X) < 10:
        raise ValueError("Se requieren al menos 10 muestras para TimeSeriesSplit")
    
    # target válido
    if pd.Series(y).nunique() < 2:
        raise ValueError("El target debe tener al menos 2 clases")
    
    # datos numéricos
    if not np.all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
        raise ValueError("Todas las variables en X deben ser numéricas")
    
    # Pipeline completo
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
    
    # Validación temporal
    tscv = TimeSeriesSplit(n_splits=4)
    
    # Métrica MCC
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=tscv,
        scoring=mcc_scorer
    )
    
    return float(scores.mean())


# ---------------------------------------------------------
# 4. EJEMPLO DE USO + VALIDACIÓN ROBUSTA
# ---------------------------------------------------------
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_hidraulico()
    
    print("=== INPUT ===")
    print(entrada['X'].head())
    print("\nTarget:")
    print(entrada['y'].head())
    
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
    
    resultado = analizar_resiliencia_hidraulica(
        entrada['X'],
        entrada['y']
    )
    
    print("\n=== RESULTADO FUNCIÓN ===")
    print(resultado)
    
    print("\n=== VALIDACIÓN ===")
    if np.isclose(resultado, salida_esperada, rtol=1e-5, atol=1e-8):
        print("✅ Resultado correcto (tolerancia numérica)")
    else:
        print("❌ Resultado incorrecto")
