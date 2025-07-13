# 🎨 Sketch to Emoji 

An interactive deep learning web app that converts hand-drawn sketches into relevant emojis using a Convolutional Neural Network (CNN). Built with TensorFlow, Flask, and deployed as a simple web app.

---

## Features

- Draw any object on the canvas (star, sun, flower, etc.)
- Recognizes 10 predefined emoji classes
- Real-time predictions using a trained CNN model
- Interactive and mobile-friendly web interface
- Ink detection to ignore blank input
- Built using TensorFlow, Flask, and NumPy

---

## Tech Stack

- Python (NumPy, TensorFlow, scikit-learn)
- Flask (Web Framework)
- HTML5 Canvas + JavaScript (Frontend sketching)
- PIL (Image handling)

---

## Setup Instructions

### 1. Clone the repository
   - git clone https://github.com/Yash6352-rs/sketch-to-emoji.git
   - cd aketch-to-emoji

### 2. Install dependencies
   - pip install -r requirements.txt or
   - pip install tensorflow flask numpy pillow scikit-learn
     
### 3. Prepare the dataset
   - python load_data.py
   - This processes and saves the dataset as X.npy and y.npy

### 4. Train the model
   - python train_cnn.py
   - Trains the CNN and saves the best model as emoji_cnn_model.h5
   
### 5. Run the web app
   - python app.py
   - Then visit http://127.0.0.1:5000 in your browser.

---

## Classes Supported

- 😊 Smiley Face
- 🌙 Moon
- ☁️ Cloud
- ⭐ Star
- ☀️ Sun
- ☂️ Umbrella
- 🌸 Flower
- 📍 Map
- 👕 T-shirt
- 🟥 Square


---

## Project Structure

### sketch-to-emoji/
   - data/(Raw and processed image data)
      - smiley_face.npy
      - ... (more .npy files)
      - X.npy / y.npy (Final training data)
   - static/ (CSS styles)
      - style.css
   - templates/
      - index.html (Web interface)
   - load_data.py  (Preprocess and load data)
   - train_cnn.py  (Train CNN model)
   - app.py  (Flask server)
   - emoji_cnn_model.h5 (Saved trained model)
   - README.md

---

📸 Preview

https://github.com/user-attachments/assets/1d220efe-f1b9-41b8-bd79-1dbafa13db78

---

## What I Learned

- This project helped me understand the full lifecycle of a deep learning application — from data preparation and model training to deployment with Flask and building an interactive frontend.

---

## Author

Created by **Yash**  
Feel free to connect on [LinkedIn](https://www.linkedin.com/in/yash6352-rs/) or check out more projects on [GitHub](https://github.com/Yash6352-rs)

