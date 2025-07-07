from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import os
import sys

# Add your project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "auto_eda_project")))

from db.db_loader import load_from_postgres
from preprocessing.cat_typo_cleaner import clean_categorical_typos
from model.train_model import train_models
from model.evaluate_model import evaluate_model
from drift.drift_detector import detect_drift_and_save
from mlflow.utils import compare_and_register_models
import joblib
import pandas as pd

# 🔧 Config
TARGET = "adjusted_total_usd"
MODEL_SAVE_PATH = "auto_eda_project/save_model/best_capstone_model.pkl"
FLASK_DEPLOY_SCRIPT = "flask/deploy_flask.sh"  # optional deployment script

# DAG definition
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 7, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    dag_id="capstone_salary_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="Automated ML pipeline for salary prediction using Airflow"
)

# ---------------------------- TASK FUNCTIONS ----------------------------

def load_data_from_db(**kwargs):
    df = load_from_postgres(table_name="software_salaries")
    df = clean_categorical_typos(df)
    kwargs['ti'].xcom_push(key="df", value=df)

def train_and_evaluate(**kwargs):
    df = kwargs['ti'].xcom_pull(key="df")
    model, X_test, y_test, run_metrics_dict = train_models(df, target=TARGET, save_path=MODEL_SAVE_PATH)
    evaluate_model(model, X_test, y_test)
    kwargs['ti'].xcom_push(key="run_metrics", value=run_metrics_dict)

def register_best_model(**kwargs):
    run_metrics_dict = kwargs['ti'].xcom_pull(key="run_metrics")
    compare_and_register_models(run_metrics_dict, model_name_prefix="BestSalaryModel")

def trigger_flask_deployment():
    if os.path.exists(FLASK_DEPLOY_SCRIPT):
        os.system(f"bash {FLASK_DEPLOY_SCRIPT}")
        print("🚀 Flask app deployment script executed.")
    else:
        print("⚠️ Flask deployment script not found.")

def detect_data_drift(**kwargs):
    drift_result = detect_drift_and_save(
        train_path="auto_eda_project/Data/train.csv",
        test_path="auto_eda_project/Data/test.csv",
        output_html_path="auto_eda_project/reports/train_vs_test_drift.html"
    )
    kwargs['ti'].xcom_push(key="drift_detected", value=drift_result.get("drift_detected", False))

def retrain_if_drift(**kwargs):
    if kwargs['ti'].xcom_pull(key="drift_detected"):
        print("🔁 Drift detected. Retraining model...")
        df = load_from_postgres(table_name="software_salaries")
        df = clean_categorical_typos(df)
        model, X_test, y_test, run_metrics_dict = train_models(df, target=TARGET, save_path=MODEL_SAVE_PATH)
        evaluate_model(model, X_test, y_test)
        compare_and_register_models(run_metrics_dict, model_name_prefix="BestSalaryModel")
    else:
        print("✅ No drift detected. Skipping retraining.")

# ---------------------------- TASKS ----------------------------

load_task = PythonOperator(
    task_id="load_data_from_postgres",
    python_callable=load_data_from_db,
    provide_context=True,
    dag=dag
)

train_task = PythonOperator(
    task_id="train_and_evaluate_model",
    python_callable=train_and_evaluate,
    provide_context=True,
    dag=dag
)

register_task = PythonOperator(
    task_id="register_best_model",
    python_callable=register_best_model,
    provide_context=True,
    dag=dag
)

deploy_task = PythonOperator(
    task_id="trigger_flask_deployment",
    python_callable=trigger_flask_deployment,
    dag=dag
)

drift_task = PythonOperator(
    task_id="detect_data_drift",
    python_callable=detect_data_drift,
    provide_context=True,
    dag=dag
)

retrain_task = PythonOperator(
    task_id="retrain_if_drift_detected",
    python_callable=retrain_if_drift,
    provide_context=True,
    dag=dag
)

# DAG Flow
load_task >> train_task >> register_task >> deploy_task >> drift_task >> retrain_task
