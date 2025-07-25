from flask import Flask, render_template, request
import numpy as np
import joblib
import json

app = Flask(__name__)

# Load model
model = joblib.load('model/rf_model.pkl')

# Load selected features
with open('model/selected_features.json') as f:
    feature_names = json.load(f)

# Load descriptive sensor labels
with open('model/sensor_labels.json') as f:
    feature_labels = json.load(f)

@app.route('/')
def home():
    return render_template('index.html',
                           feature_names=feature_names,
                           feature_labels=feature_labels,
                           form_values={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_values = request.form.to_dict()
        input_features = [float(form_values[feature]) for feature in feature_names]
        input_array = np.array([input_features])
        prediction = model.predict(input_array)[0]
        return render_template('index.html',
                               feature_names=feature_names,
                               feature_labels=feature_labels,
                               form_values=form_values,
                               prediction=f"Predicted RUL: {prediction:.2f} hours")
    except Exception as e:
        return render_template('index.html',
                               feature_names=feature_names,
                               feature_labels=feature_labels,
                               form_values=request.form.to_dict(),
                               prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
