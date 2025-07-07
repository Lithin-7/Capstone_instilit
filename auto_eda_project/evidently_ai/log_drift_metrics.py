import mlflow

def log_evidently_metrics(report_dict, prefix=""):
    """
    Logs all numeric metrics from Evidently report dict to MLflow with proper prefixes.

    Parameters:
        report_dict (dict): Output from `report.as_dict()`
        prefix (str): Prefix for metric names (e.g., 'train_val__')
    """
    for metric in report_dict.get("metrics", []):
        metric_name = metric.get("metric", "unknown")

        # 🔹 Log top-level numeric fields in "result"
        for key, value in metric.get("result", {}).items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{prefix}{metric_name}__{key}", value)

        # 🔹 Log column-wise metrics if available
        column_metrics = metric.get("result", {}).get("column_metrics", {})
        for col_name, col_result in column_metrics.items():
            for sub_key, sub_val in col_result.items():
                if isinstance(sub_val, (int, float)):
                    mlflow.log_metric(f"{prefix}{col_name}__{sub_key}", sub_val)
