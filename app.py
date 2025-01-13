from flask import Flask, request, render_template, jsonify
from flasgger import Swagger
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)
swagger = Swagger(app)  # Initialize Swagger UI

# Load the trained model
model = load_model('oral_disease_model.keras')  # Replace with your model's file path

# Define the classes
class_names = ['Caries', 'Gingivitis']  # Update this according to your classes

# Set up the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home route to display the HTML form
@app.route('/')
def home():
    return render_template('index.html')  # Make sure you have an index.html template

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    """
    This endpoint takes an image file and returns the predicted class (Caries or Gingivitis)
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The image file to classify
    responses:
      200:
        description: The predicted class and confidence
        schema:
          type: object
          properties:
            predicted_class:
              type: string
              description: The predicted class (Caries or Gingivitis)
            confidence:
              type: number
              description: The confidence level of the prediction
      400:
        description: Invalid file format or missing file
        schema:
          type: object
          properties:
            error:
              type: string
              description: Error message
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Preprocess the image and make prediction
        img = image.load_img(filename, target_size=(150, 150))  # Resize image
        img_array = image.img_to_array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image
        
        # Predict the class of the image
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        
        return jsonify({'predicted_class': predicted_class, 'confidence': float(np.max(predictions))})
    else:
        return jsonify({'error': 'Invalid file format'}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
