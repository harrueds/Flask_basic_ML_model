# Despliegue básico de un modelo de Machine Learning con Flask
### Objetivo: Implementar una API REST con Flask que permita consumir un modelo de clasificación entrenado, incorporando validación de entradas, manejo de errores y pruebas con datos en formato JSON.


Este proyecto implementa un flujo completo que comprende el entrenamiento de un modelo de clasificación utilizando scikit-learn, su serialización en un archivo local, la creación de una API REST con Flask para exponer dicho modelo, y finalmente la validación del servicio mediante un script de pruebas. El dataset utilizado es **Breast Cancer Wisconsin Diagnostic**, provisto por la librería `sklearn.datasets`.

## Estructura del proyecto

- `training.py` : Script de entrenamiento y guardado del modelo  
- `app.py` : API REST con Flask para exponer el modelo  
- `test_api.py` : Script de pruebas contra la API  
- `modelo.pkl` : Modelo entrenado (se genera al ejecutar `training.py`)  


## Requisitos

Se requiere instalar las siguientes dependencias:

* flask==3.1.2
* joblib==1.5.1
* numpy==2.0.1
* numpy-base==2.0.1
* requests==2.32.
* scikit-learn==1.7.1


## Descripción de los scripts

### Script `training.py`

Este script realiza el proceso de entrenamiento del modelo. Las principales etapas son:
1. Carga del dataset Breast Cancer desde `sklearn.datasets`.
2. División de los datos en conjuntos de entrenamiento y prueba (80%/20%).
3. Entrenamiento de un modelo de regresión logística (`LogisticRegression`).
4. Serialización y guardado del modelo en el archivo `modelo.pkl` utilizando la librería `joblib`.

Al ejecutar este script con `python training.py`, se genera el archivo `modelo.pkl` que contiene el modelo entrenado.

### Script `app.py`

Este archivo implementa una API REST utilizando Flask. Sus componentes principales son:
- Carga del modelo previamente entrenado desde `modelo.pkl`.
- Definición de dos rutas:
  - `GET /` : responde con un mensaje de estado confirmando que la API está activa.
  - `POST /predict` : recibe un JSON con la clave `features`, que debe contener una lista numérica de 30 valores correspondientes a las características del dataset. Tras la validación de los datos, la entrada se transforma en un arreglo de NumPy y se genera una predicción. La respuesta es un JSON con la clase predicha.
- Manejo de errores: ante entradas inválidas o fallos internos, la API devuelve un mensaje descriptivo y un código HTTP adecuado (400 o 500).

Para iniciar el servidor se debe ejecutar `python app.py`, lo que habilita el servicio en `http://127.0.0.1:5000/`.

### Script `test_api.py`

Este script valida el funcionamiento de la API mediante tres pruebas:
1. Petición `GET /` para comprobar que el servidor está activo.
2. Petición `POST /predict` con un conjunto válido de 30 características, que debería devolver una predicción correcta.
3. Petición `POST /predict` con un JSON mal formado, para verificar que el sistema responde con un mensaje de error coherente.

La ejecución se realiza con `python test_api.py` en una terminal separada mientras el servidor se encuentra en ejecución.

## Notas sobre el dataset

El dataset Breast Cancer Wisconsin Diagnostic contiene 30 variables numéricas que describen propiedades de núcleos celulares en imágenes digitalizadas. La variable objetivo (target) es binaria:
- 0: tumor maligno
- 1: tumor benigno

## Conclusión

El proyecto ejemplifica un flujo de trabajo básico para desplegar un modelo de aprendizaje automático. Comprende desde el entrenamiento y la serialización del modelo hasta la creación de un servicio web accesible mediante solicitudes HTTP y su validación mediante pruebas. Este enfoque constituye la base para escenarios más complejos de integración de modelos en aplicaciones reales.