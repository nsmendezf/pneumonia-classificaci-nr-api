# Clasificador de Neumonía con Deep Learning

## Descripción
API REST que predice neumonía en imágenes de rayos X usando Deep Learning.

## Modelos
- DenseNet121 (con fine tuning)
- MobileNetV2
- Ensemble learning (promedio probabilidades)

## Tecnologías
- TensorFlow
- Flask (API)
- Python

## Métricas Evaluadas
-Accuracy
-Precision
-Recall
-AUC
-Matriz de confusión

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
Predice NORMAL o PNEUMONIA con porcentaje de probabilidad.

## Deploy/Nube pública
-ngrok

### Ejemplo Uso
curl -X POST -F "file=@person1946_bacteria_4874.jpeg" \
https://theocratically-interuniversity-ranae.ngrok-free.dev/predict

### Resultado
{
  "prediction": "PNEUMONIA",
  "probability": 0.9997363686561584
}
