
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "auto_eda_project"))

# Modular imports
from data_ingestion.data_loader import load_data
from preprocessing.cat_typo_cleaner import clean_categorical_typos
from model.train_model import train_models
from model.evaluate_model import evaluate_model

# ---------- 🔧 Config ----------
USE_DB = False  # 🔁 Toggle this: True = load from DB, False = load from CSV

DATA_PATH = os.path.join("auto_eda_project", "Data", "Software_Salaries.csv")
TABLE_NAME = "software_salaries"
TARGET = "adjusted_total_usd"
MODEL_SAVE_PATH = os.path.join("auto_eda_project", "save_model", "best_capstone_model.pkl")
# -------------------------------

def main():
    print("🚀 Starting End-to-End ML Pipeline...")

    # 1️⃣ Load Dataset (from DB or CSV)
    try:
        if USE_DB:
            print(f"🔌 Loading data from PostgreSQL table: {TABLE_NAME}")
            df = load_data(from_db=True, table_name=TABLE_NAME)
        else:
            print(f"📄 Loading data from file: {DATA_PATH}")
            df = load_data(file_path=DATA_PATH)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
        # 🧼 Ensure target column is clean and numeric
    print(f"\n🔍 Target column '{TARGET}' type: {df[TARGET].dtype}")
    print(df[TARGET].describe())

    # Clean currency symbols, commas, or non-numeric issues if any
    if df[TARGET].dtype == 'object':
        print("🧽 Cleaning target column with string values...")
        df[TARGET] = (
            df[TARGET]
            .replace('[\$,₹,€,£]', '', regex=True)
            .replace(',', '', regex=True)
            .astype(float)
        )

    # Confirm cleaned target stats before transformation
    print(f"\n✅ Cleaned target stats (pre-log):\n{df[TARGET].describe()}")

    # 2️⃣ Clean typos in categorical columns
    df = clean_categorical_typos(df)

    # 3️⃣ Train & Track with MLflow + Save .pkl
    print("\n🤖 Training models and logging to MLflow...")
    model, X_test, y_test = train_models(df, target=TARGET, save_path=MODEL_SAVE_PATH)

    # 4️⃣ Final Evaluation
    print("\n📈 Evaluating best model on test set...")
    evaluate_model(model, X_test, y_test)

    print("\n✅ Pipeline completed successfully!")
    print("\n✅ DB load check completed!")
if __name__ == "__main__":
    main()



