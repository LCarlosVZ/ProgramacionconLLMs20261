import pandas as pd
import numpy as np

def generar_datos_auditoria(n_registros=100):
    """Genera el escenario aleatorio para la auditoría energética."""
    rng = np.random.default_rng(42)
    
    data = {
        'id_edificio': rng.integers(100, 500, n_registros),
        'dia_semana': rng.integers(0, 7, n_registros),
        'consumo_kWh': rng.uniform(200, 1500, n_registros),
        'ocupacion': rng.integers(10, 100, n_registros),
        'eficiencia_certificada': rng.choice(['A', 'B', 'C', 'D'], n_registros)
    }
    
    return pd.DataFrame(data)

def auditar_consumo_edificios(df):
    """Ejecuta los 5 pasos de la misión de auditoría."""
    
    # 1. Ingeniería Temporal
    df['es_fin_de_semana'] = np.where(df['dia_semana'] >= 5, 1, 0)
    
    # 2. Normalización de Ocupación
    df['consumo_por_persona'] = df['consumo_kWh'] / df['ocupacion']
    
    # promedio global ANTES del filtrado
    promedio_global = df['consumo_por_persona'].mean()
    
    # 3. Filtrado de Eficiencia (Solo A o B)
    df_filtrado = df[df['eficiencia_certificada'].isin(['A', 'B'])].copy()
    
    # 4. Cálculo de Desviación respecto al promedio global
    df_filtrado['desviacion_promedio'] = (
        df_filtrado['consumo_por_persona'] - promedio_global
    )
    
    # 5. Resultado
    return df_filtrado


# Ejemplo de uso
df_prueba = generar_datos_auditoria(50)
resultado = auditar_consumo_edificios(df_prueba)

print(resultado.head())
