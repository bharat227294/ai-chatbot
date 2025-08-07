from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import requests
import io
import tensorflow as tf
import os

app = Flask(__name__)

# Load skin cancer detection model (replace 'model_path' with actual model file path)
MODEL_PATH = "skin_cancer_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (update if different based on your model)
CLASS_NAMES = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 
               'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesion']

# Gemini API
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=YOUR_API_KEY"

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))  # Resize for model
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)

@app.route('/chat', methods=['POST'])
def chat():
    if 'image' in request.files:
        # Handle image input
        image = request.files['image'].read()
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]

        confidence = np.max(prediction)
        result = f"The uploaded image is predicted as **{predicted_class}** with confidence **{confidence:.2f}**."
        return jsonify({"reply": result})

    elif 'message' in request.json:
        # Handle text input
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        payload = {
            "contents": [
                {"parts": [{"text": user_message}]}
            ]
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, json=payload, headers=headers)
        response_data = response.json()
        bot_reply = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't process that.")
        return jsonify({"reply": bot_reply})

    else:
        return jsonify({"error": "No valid input provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
