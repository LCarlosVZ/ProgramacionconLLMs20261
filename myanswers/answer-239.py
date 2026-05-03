from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def treinar_modelo_stacking(X, y):
    """
    Configura y entrena un modelo de Stacking.
    
    Parámetros:
    - X: matriz de características (DataFrame o numpy array)
    - y: vector de etiquetas (Series o numpy array)
    
    Retorna:
    - Objeto StackingClassifier entrenado.
    """
    
    # 1. Definir los estimadores base
    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('knn', KNeighborsClassifier())
    ]
    
    # 2. Configurar el StackingClassifier con la LogisticRegression como meta-modelo
    modelo_stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression()
    )
    
    # 3. Entrenar el ensamble
    modelo_stacking.fit(X, y)
    
    # 4. Retornar el modelo entrenado
    return modelo_stacking
