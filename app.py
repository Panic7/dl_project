from io import BytesIO

from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('72_67sequential_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Read the image file
        img = image.load_img(BytesIO(file.read()), target_size=(128, 128))
        # Convert the image data to a numpy array
        img_array = image.img_to_array(img)
        # Expand the shape of the array
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocess the image data
        img_array /= 255.0
        # Use the model to make a prediction
        prediction = model.predict(img_array)[0]
        # Get the index of the highest prediction
        prediction_percentage = prediction * 100
        # Convert the index to the corresponding flower species
        labels = ['dandelion', 'daisy', 'tulip', 'sunflower', 'rose']
        # Convert the prediction probabilities to percentages
        prediction_percentage = prediction * 100
        # Create a dictionary of class labels and prediction percentages
        prediction_dict = dict(zip(labels, prediction_percentage))
        # Return the prediction
        return render_template('index.html', prediction=prediction_dict)

if __name__ == '__main__':
    app.run()