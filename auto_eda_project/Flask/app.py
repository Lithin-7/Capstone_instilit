import os
import sys
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template

# ✅ Add root path to allow relative imports (like in main.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__)

# ✅ Path to trained model
MODEL_PATH = os.path.join("..", "save_model", "best_capstone_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚫 Model not found at: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- 1️⃣ File Upload ---
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            df = pd.read_csv(file)

        # --- 2️⃣ Manual Input ---
        else:
            input_data = {key: request.form[key] for key in request.form}
            df = pd.DataFrame([input_data])

            # Attempt to convert numeric fields
            for col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass  # Leave non-numeric as-is

        # --- 3️⃣ Align Features with Model ---
        expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df.columns

        for col in expected_features:
            if col not in df.columns:
                df[col] = 0  # Default value for missing columns
        df = df[expected_features]  # Ensure correct order

        # --- 4️⃣ Predict ---
                # 4️⃣ Make Prediction
        preds = model.predict(df)
        predicted_salary = round(float(np.expm1(preds[0])), 2)  # Reverse log1p + round

        return render_template("data.html", prediction=predicted_salary)


    except Exception as e:
        return f"<h3>❌ Error: {str(e)}</h3>"

if __name__ == "__main__":
    app.run(debug=True,port=5001)
