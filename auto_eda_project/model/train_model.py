
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from preprocessing.preprocessing import get_preprocessor, log_transform_target
from auto_eda_project.mlflow.utils import start_experiment, log_model_and_metrics, compare_and_register_models


import numpy as np
import joblib

SCORING_METRIC = 'neg_mean_squared_error'

def train_models(df, target="adjusted_total_usd", save_path=None):
    # 🚀 Start MLflow experiment
    start_experiment("CAPSTONE_Salary_Experiment")

    # 🔹 Split features and target
    X = df.drop(columns=[target])
    y = log_transform_target(df[target])

    # 🧪 Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ⚙️ Get preprocessing pipeline
    preprocessor = get_preprocessor(df)

    models = {
        "DecisionTree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "regressor__max_depth": [3, 6, 5, 10]
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [3, 5, 7, 10, 15]
            }
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=42, verbosity=0),
            "params": {
                "regressor__n_estimators": [50, 100],
                "regressor__max_depth": [3, 4, 5],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__reg_alpha": [0.1, 0.5],
                "regressor__reg_lambda": [1.0]
            }
        }
    }

    best_model = None
    best_score = float("inf")
    best_name = None
    run_metrics_dict = {}

    for name, cfg in models.items():
        pipe = Pipeline([
            ('preprocessing', preprocessor),
            ('regressor', cfg['model'])
        ])

        grid = GridSearchCV(
            pipe, cfg['params'], cv=5, scoring=SCORING_METRIC, n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)

        test_neg_mse = grid.score(X_test, y_test)
        test_rmse = np.sqrt(-test_neg_mse)
        print(f"✅ {name} RMSE on Test Set: {test_rmse:.4f}")

        # 🔁 Log model and RMSE to MLflow
        run_id = log_model_and_metrics(
            model=grid.best_estimator_,
            model_name=name,
            metrics={"rmse": test_rmse},
            run_name=f"{name}_Run"
        )

        run_metrics_dict[name] = {
            "run_id": run_id,
            "rmse": test_rmse
        }

        if test_rmse < best_score:
            best_model = grid.best_estimator_
            best_score = test_rmse
            best_name = name

    print(f"\n🏆 Best Model: {best_name} | Test RMSE: {best_score:.4f}")

    # 💾 Save locally as .pkl
    if save_path:
        joblib.dump(best_model, save_path)
        print(f"📦 Saved best model locally at: {save_path}")

    # 📌 Register best model in MLflow Model Registry
    compare_and_register_models(run_metrics_dict, model_name_prefix="BestSalaryModel")

    return best_model, X_test, y_test

