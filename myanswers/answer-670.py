from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def pipeline_preprocesamiento_modelo(df, target_col):
    """
    Crea, entrena y devuelve un Pipeline con preprocesamiento 
    automático para variables numéricas y categóricas.
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identificar automáticamente columnas numéricas y categóricas
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Crear ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # 3. Crear Pipeline
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', LogisticRegression(max_iter=200))
    ])
    
    # 4. Entrenar el pipeline
    pipeline.fit(X, y)
    
    # 5. Devolver pipeline entrenado
    return pipeline
