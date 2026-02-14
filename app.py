from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.densenet import preprocess_input

app = Flask(__name__)

# Ruta principal 
@app.route("/")
def home():
    return "API Chest X-ray con Ensemble funcionando ✅"

# Cargar modelos
densenet = tf.keras.models.load_model("densenet_model.keras")
mobilenet = tf.keras.models.load_model("mobilenet_model.keras")

# Cargar clases
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = list(class_indices.keys())

# Preprocesamiento
def preprocess_image(image):
    image = image.resize((224,224))
    image = np.array(image)
    image = preprocess_input(image)   
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    img = preprocess_image(image)

    # Predicciones modelos
    pred1 = densenet.predict(img)
    pred2 = mobilenet.predict(img)

    # Ensamble
    ensemble = (pred1 + pred2) / 2
    prediction = int(ensemble > 0.5)

    return jsonify({
        "prediction": labels[prediction],
        "probability": float(ensemble[0][0])
    })

if __name__ == "__main__":
    app.run(debug=True)
