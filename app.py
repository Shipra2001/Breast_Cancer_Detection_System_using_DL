from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("breast_cancer_model.h5")

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file uploaded.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text='No selected file.')

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]
        label = 'Malignant' if prediction > 0.5 else 'Benign'
        confidence = round(float(prediction if prediction > 0.5 else 1 - prediction) * 100, 2)

        return render_template('index.html',
                               prediction_text=f"Prediction: {label}",
                               confidence_text=f"Confidence Score: {confidence}%",
                               img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
