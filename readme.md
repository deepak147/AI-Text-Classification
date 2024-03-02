## Prediction and analysis of AI Generated text using Bert and SHAP values

### Overview
This project involves building and evaluating a machine learning model for text classification. The model is trained using preprocessed text data and Google News word embeddings.

### Instructions

1. **Download Pre-trained Word Embeddings**:
   - Download the Google News word embeddings file from the following link: [Google News Word Embeddings](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)
   - Place the downloaded file in the same folder as `main.py`.

2. **Running the Model**:
   - The `main.py` file is used for building and evaluating the model.
   - Execute `main.py` to start the model building and evaluation process.

3. **Making Predictions**:
   - Use the `predict.py` file to make predictions based on the trained model.
   - This script takes a text input and outputs the model results.

4. **Running the Flask App**:
   - The Flask web application is implemented in `app.py`.
   - To run the Flask app, execute `app.py`.
   - Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser to access the app.

### File Descriptions

- `main.py`: Script for building and evaluating the model.
- `predict.py`: Script for making predictions based on the trained model.
- `app.py`: Flask web application for serving model predictions.
- `preprocess_text.py`: Module for preprocessing text data.
- `prepare_training_data.py`: Module for preparing training data.
- `build_model.py`: Module for building the machine learning model.
- `evaluate_model.py`: Module for evaluating the trained model.
- `ChatGPT Study Data.xlsx`: Excel file containing the study data.

### Dependencies
- `pickle`
- `pandas`
- `gensim`
- `Flask`
- `shap`
- `tensorflow`


Make sure to install the necessary dependencies before running the scripts or the Flask app.
