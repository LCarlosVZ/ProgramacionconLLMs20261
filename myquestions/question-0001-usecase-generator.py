import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# ---------------------------------------------------------
# 1. GENERADOR DE CASO DE USO (GROUND TRUTH) - ¡AHORA PRIMERO!
# ---------------------------------------------------------
def generar_caso_de_uso_analisis_geotecnico():
    """
    Esta función queda en la primera posición del namespace del módulo
    para que gen_funcs[0]() del evaluador la ejecute correctamente.
    """
    df = generar_casos_geotecnicos(np.random.randint(50, 150))
    umbral_nulos = np.random.uniform(0.1, 0.4)
    
    input_data = {
        'df': df.copy(),
        'umbral_nulos': umbral_nulos
    }
    
    # -------- GROUND TRUTH --------
    df_limpio = df.copy()
    
    porcentaje_nulos = df_limpio.isnull().mean()
    columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > umbral_nulos].index
    df_limpio = df_limpio.drop(columns=columnas_a_eliminar)
    
    columnas_numericas = df_limpio.select_dtypes(include=np.number).columns
    
    for col in columnas_numericas:
        if df_limpio[col].isnull().any():
            if df_limpio[col].dropna().empty:
                continue
            moda = df_limpio[col].mode()[0]
            df_limpio[col] = df_limpio[col].fillna(moda)
    
    if 'estabilidad' not in df_limpio.columns:
        raise ValueError("La columna 'estabilidad' fue eliminada.")
    
    df_limpio['estabilidad_num'] = df_limpio['estabilidad'].map({
        'Inestable': 0,
        'Estable': 1
    })
    
    columnas_agg = {}

    if 'pluviosidad_reciente' in df_limpio.columns:
        columnas_agg['pluviosidad_reciente'] = 'max'

    if 'presion_de_poros' in df_limpio.columns:
        columnas_agg['presion_de_poros'] = 'mean'

    if not columnas_agg:
        raise ValueError("No hay columnas para agrupar.")
    
    output_data = df_limpio.groupby('estabilidad_num').agg(columnas_agg)
    output_data = output_data.sort_index().sort_index(axis=1)
    
    return input_data, output_data


# ---------------------------------------------------------
# 2. GENERADOR DE DATOS ALEATORIOS
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 3. FUNCIÓN PRINCIPAL (SOLUCIÓN)
# ---------------------------------------------------------
def analizar_umbrales_geotecnicos(df, umbral_nulos):
    df_limpio = df.copy()
    
    # 1. Eliminación de columnas con muchos nulos
    porcentaje_nulos = df_limpio.isnull().mean()
    columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > umbral_nulos].index
    df_limpio = df_limpio.drop(columns=columnas_a_eliminar)
    
    # 2. Imputación con moda (robusta)
    columnas_numericas = df_limpio.select_dtypes(include=np.number).columns
    
    for col in columnas_numericas:
        if df_limpio[col].isnull().any():
            if df_limpio[col].dropna().empty:
                continue
            moda = df_limpio[col].mode()[0]
            df_limpio[col] = df_limpio[col].fillna(moda)
    
    # 3. Codificación
    if 'estabilidad' not in df_limpio.columns:
        raise ValueError("La columna 'estabilidad' fue eliminada.")
    
    df_limpio['estabilidad_num'] = df_limpio['estabilidad'].map({
        'Inestable': 0,
        'Estable': 1
    })
    
    # 4. Aggreg dinámico
    columnas_agg = {}

    if 'pluviosidad_reciente' in df_limpio.columns:
        columnas_agg['pluviosidad_reciente'] = 'max'

    if 'presion_de_poros' in df_limpio.columns:
        columnas_agg['presion_de_poros'] = 'mean'

    if not columnas_agg:
        raise ValueError("No hay columnas válidas para agrupar.")
    
    df_agrupado = df_limpio.groupby('estabilidad_num').agg(columnas_agg)
    
    # Ordenar índice y columnas (CLAVE para comparar)
    df_agrupado = df_agrupado.sort_index().sort_index(axis=1)
    
    return df_agrupado


# ---------------------------------------------------------
# 4. PRUEBA DE INTEGRACIÓN LOCAL
# ---------------------------------------------------------
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_analisis_geotecnico()
    
    print("=== INPUT ===")
    print(entrada['df'].head())
    print("\nUmbral:", entrada['umbral_nulos'])
    
    resultado = analizar_umbrales_geotecnicos(
        entrada['df'],
        entrada['umbral_nulos']
    )
    
    print("\n=== VALIDACIÓN ROBUSTA ===")
    try:
        assert_frame_equal(
            resultado,
            salida_esperada,
            check_dtype=False,
            rtol=1e-5,
            atol=1e-8
        )
        print("✅ Resultado correcto en el entorno local.")
    except AssertionError as e:
        print("❌ Error en la comparación:")
        print(e)
