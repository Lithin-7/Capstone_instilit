import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def start_experiment(experiment_name="CAPSTONE_Salary_Experiment"):
    """
    Initializes an MLflow experiment and enables sklearn autologging.
    """
    mlflow.set_experiment(experiment_name)
    #mlflow.sklearn.autolog()
    print(f"🧪 Tracking experiment: {experiment_name}")


def log_model_and_metrics(model, model_name, metrics: dict, run_name="ModelRun"):
    """
    Logs a trained model and its metrics to MLflow.
    
    Parameters:
        model: Trained sklearn model
        model_name (str): Name to store model under in artifacts
        metrics (dict): Dictionary of evaluation metrics (e.g., {'rmse': 1.2, 'r2': 0.9})
        run_name (str): MLflow run name
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.sklearn.log_model(sk_model=model, artifact_path=model_name)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        run_id = run.info.run_id
        print(f"📌 Model logged to run: {run_id}")
        return run_id


def compare_and_register_models(run_metrics_dict, model_name_prefix="BestSalaryModel"):
    """
    Compares multiple MLflow runs and registers the best model based on RMSE (lower is better).
    
    Parameters:
        run_metrics_dict (dict): {
            "ModelName": {
                "run_id": "<run_id>",
                "rmse": <float>
            }, ...
        }
        model_name_prefix (str): Prefix for registered model name

    Returns:
        str: Registered model name
    """
    # ✅ Select model with lowest RMSE
    best_model = min(run_metrics_dict.items(), key=lambda x: x[1]["rmse"])
    best_name, best_info = best_model
    best_run_id = best_info["run_id"]

    print(f"\n🥇 Best Model: {best_name} with RMSE: {best_info['rmse']:.4f}")

    model_uri = f"runs:/{best_run_id}/{best_name}"
    registry_name = f"{model_name_prefix}_{best_name}"
    mlflow.register_model(model_uri=model_uri, name=registry_name)
    print(f"✅ Registered as: {registry_name}")

    return registry_name
