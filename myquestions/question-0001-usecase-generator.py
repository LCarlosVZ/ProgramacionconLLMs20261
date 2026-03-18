import pandas as pd
import numpy as np

def generar_casos_geotecnicos(n_muestras=100):
    # Configuramos el generador aleatorio
    rng = np.random.default_rng(42)
    
    # 1. Creamos la estructura base 
    data = {
        'sensor_id': rng.integers(1000, 2000, n_muestras),
        'pluviosidad_reciente': rng.uniform(0, 150, n_muestras),
        'presion_de_poros': rng.uniform(20, 100, n_muestras),
        'inclinacion_gradual': rng.uniform(5, 45, n_muestras),
        'estabilidad': rng.choice(['Estable', 'Inestable'], n_muestras),
        'sensor_auxiliar_A': rng.uniform(0, 10, n_muestras),
        'sensor_auxiliar_B': rng.uniform(0, 10, n_muestras)
    }
    df = pd.DataFrame(data)

    # 2. Inyectamos "problemas" (Valores Nulos)
    # Sensor B: Ponemos 30% de nulos (esto hará que la función lo ELIMINE)
    df.loc[rng.choice(df.index, int(n_muestras * 0.30), replace=False), 'sensor_auxiliar_B'] = np.nan
    
    # Pluviosidad: Ponemos 5% de nulos (esto hará que la función los IMPUTE con la moda)
    df.loc[rng.choice(df.index, int(n_muestras * 0.05), replace=False), 'pluviosidad_reciente'] = np.nan

    return df
