import numpy as np
import pandas as pd


# =====================================================================
# 1. FUNCIÓN PRINCIPAL (El calificador ejecutará esta por estar de primera)
# =====================================================================
def generar_caso_de_uso_auditoria():
    """Genera un caso de prueba aleatorio (input y output esperado)"""

    # 1. Datos aleatorios usando la función auxiliar de abajo
    n = np.random.randint(50, 150)
    df = generar_datos_auditoria(n)

    # 2. INPUT
    input_data = {"df": df.copy()}

    df_gt = df.copy()

    # Ingeniería temporal
    df_gt["es_fin_de_semana"] = np.where(df_gt["dia_semana"] >= 5, 1, 0)

    # Manejo de división por cero
    df_gt["consumo_por_persona"] = df_gt["consumo_kWh"] / df_gt[
        "ocupacion"
    ].replace(0, np.nan)

    # Promedio global
    promedio_global = df_gt["consumo_por_persona"].mean()

    # Filtrado
    df_filtrado = df_gt[
        df_gt["eficiencia_certificada"].isin(["A", "B"])
    ].copy()

    # Desviación
    df_filtrado["desviacion_promedio"] = (
        df_filtrado["consumo_por_persona"] - promedio_global
    )

    # El output esperado que exige el calificador
    output_data = (df_filtrado, promedio_global)

    return input_data, output_data


# =====================================================================
# 2. FUNCIONES AUXILIARES Y DE REFERENCIA (Abajo para que no interfieran)
# =====================================================================
def generar_datos_auditoria(n_registros=100):
    """Genera el escenario aleatorio para la auditoría energética."""
    rng = np.random.default_rng()

    data = {
        "id_edificio": rng.integers(100, 500, n_registros),
        "dia_semana": rng.integers(0, 7, n_registros),
        "consumo_kWh": rng.uniform(200, 1500, n_registros),
        "ocupacion": rng.integers(0, 100, n_registros),  # Puede haber ceros
        "eficiencia_certificada": rng.choice(["A", "B", "C", "D"], n_registros),
    }

    return pd.DataFrame(data)


def auditar_consumo_edificios(df):
    """Ejecuta los 5 pasos de la misión de auditoría."""
    df = df.copy()

    # 1. Ingeniería Temporal
    df["es_fin_de_semana"] = np.where(df["dia_semana"] >= 5, 1, 0)

    # 2. Normalización
    df["consumo_por_persona"] = df["consumo_kWh"] / df["ocupacion"].replace(
        0, np.nan
    )

    # 3. Promedio global
    promedio_global = df["consumo_por_persona"].mean()

    # 4. Filtrado
    df_filtrado = df[df["eficiencia_certificada"].isin(["A", "B"])].copy()

    # 5. Desviación
    df_filtrado["desviacion_promedio"] = (
        df_filtrado["consumo_por_persona"] - promedio_global
    )

    return df_filtrado, promedio_global


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_auditoria()

    print("=== INPUT ===")
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO (DATAFRAME) ===")
    print(salida_esperada[0].head())
    print(f"Promedio Global Esperado: {salida_esperada[1]}")
