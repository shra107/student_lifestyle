from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and feature columns
model = joblib.load("trained_depression_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        input_dict = {
            "Age": float(request.form['age']),
            "Gender": request.form['gender'],
            "Department": request.form['department'],
            "CGPA": float(request.form['cgpa']),
            "Sleep_Duration": float(request.form['sleep']),
            "Study_Hours": float(request.form['study_hours']),
            "Social_Media_Hours": float(request.form['social_media']),
            "Physical_Activity": float(request.form['physical_activity']),
            "Stress_Level": float(request.form['stress'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # One-hot encode
        input_encoded = pd.get_dummies(input_df)

        # Match training columns
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Scale
        scaled_input = scaler.transform(input_encoded)

        # Predict
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        if prediction == 1:
            result = f"⚠ Depression Risk Detected (Confidence: {probability*100:.2f}%)"
        else:
            result = f"✅ No Depression Risk (Confidence: {(1-probability)*100:.2f}%)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html",
                               prediction_text="Invalid Input! Please check all fields.")

if __name__ == "__main__":
    app.run(debug=True)