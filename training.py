"""
training.py

Este script implementa el proceso de entrenamiento y serialización de un modelo de clasificación binaria
utilizando el dataset Breast Cancer Wisconsin Diagnostic provisto por la librería scikit-learn. El objetivo
principal es generar un modelo capaz de distinguir entre tumores malignos y benignos en función de un conjunto
de 30 características numéricas extraídas de imágenes digitalizadas de núcleos celulares.

Flujo general del script:
1. Carga del dataset: se obtiene tanto la matriz de características (X) como el vector objetivo (y).
2. División de datos: los registros se reparten en conjuntos de entrenamiento y prueba, garantizando
   reproducibilidad mediante una semilla aleatoria fija.
3. Definición y entrenamiento del modelo: se emplea una regresión logística, ampliamente utilizada en
   problemas de clasificación binaria debido a su interpretabilidad y robustez. Se incrementa el número
   máximo de iteraciones para asegurar la convergencia.
4. Serialización del modelo: el objeto entrenado se guarda en disco con el nombre 'modelo.pkl' utilizando
   la librería joblib, lo que permite su reutilización sin necesidad de reentrenar en ejecuciones futuras.
5. Confirmación en consola: se informa al usuario que el modelo ha sido entrenado y almacenado exitosamente.

Este script constituye el primer paso de un flujo de despliegue de Machine Learning, donde el modelo
posteriormente será consumido a través de una API REST implementada en Flask (app.py).
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Cargar dataset
X, y = load_breast_cancer(return_X_y=True)

# 2. Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Entrenar modelo
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# 4. Guardar modelo
joblib.dump(model, "modelo.pkl")

# 5. Confirmación
print("Modelo entrenado y guardado como modelo.pkl")
