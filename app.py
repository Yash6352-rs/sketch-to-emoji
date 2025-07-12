import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the trained model
model = load_model('emoji_cnn_model.h5')

# Class labels and emoji mapping
class_labels = [
    'smiley_face',
    'moon',
    'map',
    'cloud',
    'star',
    'sun',
    'umbrella',
    'flower',
    't-shirt',
    'square'
]

emoji_map = {
    'smiley_face': '😊',
    'moon': '🌙',
    'map': '📍',
    'cloud': '☁️',
    'star': '⭐',
    'sun': '🌞',
    'umbrella': '☂️',
    'flower': '🌸',
    't-shirt': '👕',
    'square': '🟥'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image'].split(",")[1] 
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))  
        image = Image.alpha_composite(bg, image).convert('L') 

        image = image.resize((28, 28))
        img_array = np.array(image).astype("float32") / 255.0

        ink_intensity = img_array.sum()
        print("Ink intensity:", round(ink_intensity, 2))

        # Check for blank images
        if ink_intensity < 5:
            print("Blank or nearly blank input detected.")
            return jsonify({"error": "Blank image", "emoji": None})

        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_index]
        confidence = predictions[0][predicted_index]

        emoji = emoji_map.get(predicted_label, "❓")

        print(f"Predicted: {predicted_label} -> {emoji} (confidence: {confidence:.2f})")

        return jsonify({
            "prediction": predicted_label,
            "emoji": emoji,
            "confidence": round(float(confidence), 2)
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Prediction failed", "emoji": None})

if __name__ == '__main__':
    app.run(debug=True)
