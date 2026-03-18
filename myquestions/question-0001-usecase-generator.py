import pandas as pd
import numpy as np

def generar_casos_geotecnicos(n_muestras=100):
    rng = np.random.default_rng(42)
    
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
    
    # 1. Limpieza Dinámica
    porcentaje_nulos = df.isnull().mean()
    columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > umbral_nulos].index
    df_limpio = df.drop(columns=columnas_a_eliminar)
    
    # 2. Imputación de Tendencia (moda en columnas numéricas)
    columnas_numericas = df_limpio.select_dtypes(include=np.number).columns
    
    for col in columnas_numericas:
        moda = df_limpio[col].mode()[0]
        df_limpio[col] = df_limpio[col].fillna(moda)
    
    # 3. Codificación de Seguridad
    df_limpio['estabilidad_num'] = df_limpio['estabilidad'].map({
        'Inestable': 0,
        'Estable': 1
    })
    
    # 4. Resumen Estadístico
    df_agrupado = df_limpio.groupby('estabilidad_num').agg({
        'pluviosidad_reciente': 'max',
        'presion_de_poros': 'mean'
    })
    
    # 5. Resultado
    return df_agrupado


# Ejemplo de uso
df_prueba = generar_casos_geotecnicos(100)
resultado = analizar_umbrales_geotecnicos(df_prueba, 0.2)

print(resultado)
