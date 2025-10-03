"""
app.py

Este script implementa una API REST utilizando Flask para exponer un modelo de Machine Learning previamente
entrenado y serializado. El modelo corresponde a una regresión logística entrenada con el dataset
Breast Cancer Wisconsin Diagnostic. La finalidad de este servicio es recibir instancias de entrada en formato
JSON, procesarlas y devolver una predicción binaria que clasifica un tumor como maligno o benigno.

Flujo general del script:
1. Carga del modelo: se importa desde el archivo 'modelo.pkl', previamente generado por training.py,
   evitando la necesidad de reentrenamiento en cada ejecución.
2. Configuración de Flask: se inicializa la aplicación web, que funcionará como servidor local para atender
   peticiones HTTP.
3. Definición de endpoints:
   - GET / : ruta de verificación que confirma la disponibilidad del servicio devolviendo un mensaje de estado.
   - POST /predict : endpoint principal que recibe un JSON con la clave "features". El valor asociado debe ser
     una lista numérica de 30 elementos, correspondientes a las características del dataset. El endpoint valida
     la estructura de los datos, transforma la lista en un arreglo de NumPy con la dimensión apropiada, genera
     una predicción con el modelo cargado y devuelve el resultado en formato JSON.
4. Manejo de errores: se incluyen validaciones para entradas faltantes o mal estructuradas, devolviendo un
   código de estado HTTP 400. Ante errores inesperados, el servicio responde con un mensaje de error y código 500,
   lo que facilita la depuración durante el desarrollo.

Este script constituye la pieza central del despliegue, permitiendo que el modelo de clasificación pueda ser
consumido por clientes externos a través de solicitudes HTTP estándar.
"""

# Paquetes necesarios
# pip install flask joblib scikit-learn numpy

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Inicializar aplicación Flask
app = Flask(__name__)

# 1. Cargar modelo entrenado
model = joblib.load("modelo.pkl")


# 2. Endpoint raíz: verificación de estado
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API lista"}), 200


# 3. Endpoint de predicción
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # a) Extraer JSON del cuerpo de la petición
        data = request.get_json()
        if "features" not in data:
            return jsonify({"Error 400": "Falta la clave 'features' en el JSON"}), 400

        features = data["features"]

        # b) Validar que 'features' sea una lista
        if not isinstance(features, list):
            return jsonify({"Error 400": "'features' debe ser una lista"}), 400

        # c) Convertir a array NumPy y ajustar dimensiones
        X = np.array(features).reshape(1, -1)

        # d) Generar predicción
        pred = model.predict(X)[0]

        # e) Devolver resultado como JSON
        return jsonify({"Prediction": int(pred)}), 200

    except Exception as e:
        # Manejo de errores inesperados
        return jsonify({"Error 500": str(e)}), 500


# 4. Ejecución del servidor en modo desarrollo
if __name__ == "__main__":
    app.run(debug=True)
