from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)
predictor = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        prediction = predictor.predict(df)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})