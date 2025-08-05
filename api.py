from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ Fraud Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        expected_cols = [
            'TotalReimbursed', 'AvgReimbursed', 'NumClaims',
            'TotalDeductibles', 'UniquePatients',
            'NumClaimIDs', 'AvgLengthOfStay'
        ]
        for col in expected_cols:
            if col not in df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        df = df[expected_cols]
        prediction = model.predict(df)[0]
        result = "Yes" if prediction == 1 else "No"
        return jsonify({"PotentialFraud": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
