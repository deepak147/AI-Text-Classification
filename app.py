import pickle

import numpy as np

from predict import predict
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictt', methods=['POST'])
def predictt():
    
    with open("label_encoder.pkl", "rb") as file: 
        label_encoder = pickle.load(file)
    text = request.form['text']
    prediction_probabilities = predict(text)
    predicted_labels = label_encoder.inverse_transform(np.argmax(prediction_probabilities, axis=1))
    max_probability = np.around(100*(np.amax(prediction_probabilities)), decimals=2)
    return render_template('result.html', text=text, predicted_class=predicted_labels[0], probability=max_probability)

if __name__ == '__main__':
    app.run(debug=True)
