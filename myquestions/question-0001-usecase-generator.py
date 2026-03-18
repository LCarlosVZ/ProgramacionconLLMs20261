import pandas as pd
import numpy as np

def generar_casos_geotecnicos(n_muestras=100):
    rng = np.random.default_rng()
    
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

    # Inyectar nulos
    df.loc[rng.choice(df.index, int(n_muestras * 0.30), replace=False), 'sensor_auxiliar_B'] = np.nan
    df.loc[rng.choice(df.index, int(n_muestras * 0.05), replace=False), 'pluviosidad_reciente'] = np.nan

    return df


def analizar_umbrales_geotecnicos(df, umbral_nulos):
    
    # Trabajar sobre copia
    df_limpio = df.copy()
    
    # 1. Eliminación de columnas con muchos nulos
    porcentaje_nulos = df_limpio.isnull().mean()
    columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > umbral_nulos].index
    df_limpio = df_limpio.drop(columns=columnas_a_eliminar)
    
    # 2. Imputación con moda
    columnas_numericas = df_limpio.select_dtypes(include=np.number).columns
    
    for col in columnas_numericas:
        if df_limpio[col].isnull().any():
            moda = df_limpio[col].mode()[0]
            df_limpio[col] = df_limpio[col].fillna(moda)
    
    # 3. Codificación
    df_limpio['estabilidad_num'] = df_limpio['estabilidad'].map({
        'Inestable': 0,
        'Estable': 1
    })
    
    # 4. Agrupación
    df_agrupado = df_limpio.groupby('estabilidad_num').agg({
        'pluviosidad_reciente': 'max',
        'presion_de_poros': 'mean'
    })
    
    return df_agrupado


# FUNCIÓN GENERADORA 
def generar_caso_de_uso_analisis_geotecnico():
    """
    Genera input y output esperado (ground truth independiente)
    """
    
    df = generar_casos_geotecnicos(np.random.randint(50, 150))
    umbral_nulos = np.random.uniform(0.1, 0.4)
    
    # INPUT
    input_data = {
        'df': df.copy(),
        'umbral_nulos': umbral_nulos
    }
    
    df_limpio = df.copy()
    
    # 1. Eliminación de columnas
    porcentaje_nulos = df_limpio.isnull().mean()
    columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > umbral_nulos].index
    df_limpio = df_limpio.drop(columns=columnas_a_eliminar)
    
    # 2. Imputación
    columnas_numericas = df_limpio.select_dtypes(include=np.number).columns
    
    for col in columnas_numericas:
        if df_limpio[col].isnull().any():
            moda = df_limpio[col].mode()[0]
            df_limpio[col] = df_limpio[col].fillna(moda)
    
    # 3. Codificación
    df_limpio['estabilidad_num'] = df_limpio['estabilidad'].map({
        'Inestable': 0,
        'Estable': 1
    })
    
    # 4. Agrupación
    output_data = df_limpio.groupby('estabilidad_num').agg({
        'pluviosidad_reciente': 'max',
        'presion_de_poros': 'mean'
    })
    
    return input_data, output_data
