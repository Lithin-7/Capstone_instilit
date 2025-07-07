import os
import sys
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset, DataDriftPreset, DataQualityPreset
from sklearn.model_selection import train_test_split
from functools import reduce

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from auto_eda_project.evidently_ai.log_drift_metrics import log_evidently_metrics


def run_evidently_drift():
    mlflow.set_experiment("Data Drift Monitoring")

    # Load historical data
    historical = pd.read_csv("auto_eda_project/Data/Software_Salaries.csv")

    # Drop initially known fully empty columns
    columns_to_drop = ["education", "skills"]
    historical.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Train/Val/Test Split
    X = historical.drop(columns=["adjusted_total_usd"])
    y = historical["adjusted_total_usd"]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Detect and drop fully empty columns in any split
    all_dfs = [X_train, X_val, X_test]
    empty_cols_in_any_split = reduce(
        lambda acc, df: acc.union(set(df.columns[df.isna().all()])),
        all_dfs,
        set()
    )
    if empty_cols_in_any_split:
        print(f"🧹 Dropping completely empty columns in any split: {empty_cols_in_any_split}")
        X_train.drop(columns=empty_cols_in_any_split, inplace=True)
        X_val.drop(columns=empty_cols_in_any_split, inplace=True)
        X_test.drop(columns=empty_cols_in_any_split, inplace=True)

    # ------------------ 1️⃣ Train vs Validation ------------------
    with mlflow.start_run(run_name="Drift_Train_vs_Validation"):
        train_df = X_train.copy()
        train_df["target"] = y_train
        val_df = X_val.copy()
        val_df["target"] = y_val

        report = Report(metrics=[TargetDriftPreset(), DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=train_df, current_data=val_df)
        report.save_html("auto_eda_project/drift_reports/train_vs_validation_drift.html")
        mlflow.log_artifact("train_vs_validation_drift.html")

        results = report.as_dict()
        log_evidently_metrics(results, prefix="train_val__")

    # ------------------ 2️⃣ Validation vs Test ------------------
    with mlflow.start_run(run_name="Drift_Validation_vs_Test"):
        val_df = X_val.copy()
        val_df["target"] = y_val
        test_df = X_test.copy()
        test_df["target"] = y_test

        report = Report(metrics=[TargetDriftPreset(), DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=val_df, current_data=test_df)
        report.save_html("auto_eda_project/drift_reports/validation_vs_test_drift.html")
        mlflow.log_artifact("validation_vs_test_drift.html")

        results = report.as_dict()
        log_evidently_metrics(results, prefix="val_test__")

    # ------------------ 3️⃣ Historical vs New ------------------
    # Commented for now (new_data not available)
    """
    new_data = pd.read_csv("auto_eda_project/Data/New_data.csv")
    new_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    with mlflow.start_run(run_name="Drift_Historical_vs_NewData"):
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=X, current_data=new_data)
        report.save_html("historical_vs_newdata_drift.html")
        mlflow.log_artifact("historical_vs_newdata_drift.html")

        results = report.as_dict()
        log_evidently_metrics(results, prefix="historic__")
    """

    print("✅ Drift monitoring for Train/Val/Test complete.")


if __name__ == "__main__":
    run_evidently_drift()
