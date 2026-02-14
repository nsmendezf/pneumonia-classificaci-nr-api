# Clasificador de Neumonía con Deep Learning

## Descripción
API REST que predice neumonía en imágenes de rayos X usando Deep Learning.

## Modelos
- DenseNet121 (fine tuning)
- MobileNetV2
- Ensemble learning

## Tecnologías
- TensorFlow
- Flask
- Python

## Ejecutar proyecto

### 1. Crear entorno
python -m venv venv
source venv/bin/activate

### 2. Instalar dependencias
pip install -r requirements.txt

### 3. Ejecutar API
python app.py

## Endpoint

POST /predict

Envía imagen y retorna predicción.

## Resultado
Predice NORMAL o PNEUMONIA con probabilidad.
