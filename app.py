from flask import Flask, jsonify, request
from flask_cors import CORS
from keras._tf_keras.keras.models import load_model
import cv2
import numpy as np
import urllib.request
import os
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load saved model
model = load_model('models/LumpyDisease.h5')

def isInfected(img_path):
    

    # Your existing function
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array.shape
    img_array = img_array.reshape(1, 150, 150, 3)
    a = model.predict(img_array)
    indices = a.argmax()
    confidence = np.max(a) * 100
    if indices == 0:
        isCattleInfected = False
    else:
        isCattleInfected = True
    return isCattleInfected, confidence

@app.route('/predict', methods=['GET'])
def predict():
    
    image_url = request.args.get('image_url')
    image_url = image_url.strip('"')  # Remove quotation marks
    
    img_path = 'temp_image.jpg'
    
    try:
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            # Download the image and save it to a temporary file
            with open(img_path, 'wb') as f:
                f.write(response.content)
            
            # Make a prediction using the model
            isCattleInfected, confidence = isInfected(img_path)
            
            # Remove the temporary file
            os.remove(img_path)
            
            # Return the result as a JSON response
            return jsonify({
                'isInfected': isCattleInfected,
                'confidence': f'{confidence:.2f}%',
                'error': "none"
            })
        else:
            # Return an error message if the image could not be retrieved
            return jsonify({
                'isInfected': 'none',
                'confidence': 'none',
                'error': 'Failed to retrieve image'}), 500
    except requests.exceptions.RequestException as e:
        # Return an error message if there was an error with the request
        return jsonify({
            'isInfected': 'none',
            'confidence': 'none',
            'error': f'Image retrieval failed'}), 500

@app.route('/')
def hello_world():
    return 'Cattle health Api'

