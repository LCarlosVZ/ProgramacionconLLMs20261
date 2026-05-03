from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def clasificar_knn(X, y, nuevo_punto):
    """
    Clasifica un nuevo punto utilizando el algoritmo KNN.
    
    Parámetros:
    - X: matriz de características (n_samples, n_features)
    - y: vector de etiquetas (n_samples,)
    - nuevo_punto: dato a clasificar (1, n_features)
    
    Retorna:
    - Predicción de la clase para el nuevo_punto.
    """
    
    # 1. Instanciar el modelo (se utiliza n_neighbors=5 por defecto)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # 2. Entrenar el modelo con los datos proporcionados
    knn.fit(X, y)
    
    # 3. Predecir la clase del nuevo_punto
    # reshape(1, -1) asegura que el punto tenga la forma correcta para la predicción
    prediccion = knn.predict(nuevo_punto.reshape(1, -1))
    
    # 4. Devolver la predicción
    return prediccion[0]

# Ejemplo de uso integrado con tu generador:
# datos = generar_caso_de_uso_clasificar_knn()
# resultado = clasificar_knn(datos['X'], datos['y'], datos['nuevo_punto'])
# print(f"La clase predicha es: {resultado}")
