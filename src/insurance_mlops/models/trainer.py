"""Model training, cross-validation comparison, and MLflow experiment tracking."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

from insurance_mlops.utils.common import get_project_root, load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """Calculates standard regression evaluation metrics (MAE, RMSE, R2).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary containing MAE, RMSE, and R2.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def get_model_candidates(random_seed: int = 42) -> Dict[str, Any]:
    """Returns the candidate regression models with reproducible seeds.

    Args:
        random_seed: Seed for random state.

    Returns:
        Dictionary mapping model names to regressor instances.
    """
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            random_state=random_seed,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=4,
            random_state=random_seed,
        ),
    }


def cross_validate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Performs K-Fold Cross Validation strictly on the training partition.

    Args:
        model: Regressor instance to evaluate.
        X_train: Preprocessed training features.
        y_train: Training target values (1D array).
        cv_folds: Number of cross-validation folds.
        random_seed: Seed for KFold shuffle.

    Returns:
        Dictionary of mean and standard deviation for RMSE, MAE, and R2 across folds.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    rmses: List[float] = []
    maes: List[float] = []
    r2s: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_f_train, X_f_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_f_train, y_f_val = y_train[train_idx], y_train[val_idx]

        # Clone or re-fit on fold
        fold_model = joblib.load(joblib.dump(model, "/tmp/temp_model.joblib")[0])
        fold_model.fit(X_f_train, y_f_train)
        preds = fold_model.predict(X_f_val)

        metrics = calculate_metrics(y_f_val, preds)
        rmses.append(metrics["rmse"])
        maes.append(metrics["mae"])
        r2s.append(metrics["r2"])

    return {
        "cv_rmse": round(float(np.mean(rmses)), 4),
        "cv_rmse_std": round(float(np.std(rmses)), 4),
        "cv_mae": round(float(np.mean(maes)), 4),
        "cv_mae_std": round(float(np.std(maes)), 4),
        "cv_r2": round(float(np.mean(r2s)), 4),
        "cv_r2_std": round(float(np.std(r2s)), 4),
    }


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: Optional[Dict[str, Any]] = None,
    cv_folds: int = 5,
    random_seed: int = 42,
    enable_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any, str, Dict[str, Any]]:
    """Trains and compares all candidate models using cross-validation on training data.

    Logs each model, parameters, and metrics to MLflow, and selects the
    best-performing model based on the lowest cross-validation RMSE.

    Args:
        X_train: Preprocessed training features DataFrame.
        y_train: Training target Series.
        candidates: Optional dictionary of candidate models.
        cv_folds: Number of folds for cross-validation.
        random_seed: Random seed for reproducibility.
        enable_mlflow: Whether to log to MLflow.
        tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.

    Returns:
        Tuple of (comparison_df, best_model, best_model_name, best_model_metrics).
    """
    logger.info("Starting model training and cross-validation comparison...")
    if candidates is None:
        candidates = get_model_candidates(random_seed=random_seed)

    y_train_arr = y_train.to_numpy().ravel()
    results_list: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Any] = {}

    if enable_mlflow:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    for model_name, model in candidates.items():
        logger.info(f"--- Evaluating Candidate: {model_name} ---")

        # 1. K-Fold Cross-Validation on Training Data Only (Zero Test Leakage)
        cv_metrics = cross_validate_model(
            model=model,
            X_train=X_train,
            y_train=y_train_arr,
            cv_folds=cv_folds,
            random_seed=random_seed,
        )
        logger.info(
            f"[{model_name}] CV Results: RMSE={cv_metrics['cv_rmse']} ± {cv_metrics['cv_rmse_std']}, "
            f"MAE={cv_metrics['cv_mae']}, R2={cv_metrics['cv_r2']}"
        )

        # 2. Fit model on full training set
        model.fit(X_train, y_train_arr)
        train_preds = model.predict(X_train)
        train_metrics = calculate_metrics(y_train_arr, train_preds)
        fitted_models[model_name] = model

        # Combined summary record
        record = {
            "model_name": model_name,
            "cv_rmse": cv_metrics["cv_rmse"],
            "cv_rmse_std": cv_metrics["cv_rmse_std"],
            "cv_mae": cv_metrics["cv_mae"],
            "cv_r2": cv_metrics["cv_r2"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
        }
        results_list.append(record)

        # 3. Log to MLflow
        if enable_mlflow:
            try:
                with mlflow.start_run(run_name=f"candidate_{model_name}") as run:
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("random_seed", random_seed)
                    mlflow.log_param("cv_folds", cv_folds)
                    mlflow.log_params(model.get_params())

                    # Log CV & train metrics
                    mlflow.log_metric("cv_rmse", cv_metrics["cv_rmse"])
                    mlflow.log_metric("cv_rmse_std", cv_metrics["cv_rmse_std"])
                    mlflow.log_metric("cv_mae", cv_metrics["cv_mae"])
                    mlflow.log_metric("cv_r2", cv_metrics["cv_r2"])
                    mlflow.log_metric("train_rmse", train_metrics["rmse"])
                    mlflow.log_metric("train_mae", train_metrics["mae"])
                    mlflow.log_metric("train_r2", train_metrics["r2"])

                    mlflow.set_tag("stage", "candidate_comparison")
                    mlflow.set_tag("framework", "scikit-learn")

                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name="model",
                        skops_trusted_types=["sklearn.tree._tree.Tree"],
                    )
                    logger.info(f"[{model_name}] Logged run to MLflow (run_id={run.info.run_id})")
            except Exception as e:
                logger.warning(f"MLflow logging encountered an issue for {model_name}: {e}")

    # Convert to DataFrame and sort by lowest CV RMSE
    comparison_df = pd.DataFrame(results_list).sort_values(by="cv_rmse", ascending=True).reset_index(drop=True)

    best_record = comparison_df.iloc[0].to_dict()
    best_model_name = str(best_record["model_name"])
    best_model = fitted_models[best_model_name]

    logger.info(
        f"★ Selected Best Model: '{best_model_name}' with lowest "
        f"CV RMSE={best_record['cv_rmse']} (R2={best_record['cv_r2']})"
    )

    # Log best model tag to MLflow if enabled
    if enable_mlflow:
        try:
            with mlflow.start_run(run_name=f"best_{best_model_name}") as run:
                mlflow.log_param("selected_best_model", best_model_name)
                for k, v in best_record.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"best_{k}", v)
                mlflow.set_tag("is_best", "true")
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="best_model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )
        except Exception as e:
            logger.warning(f"MLflow best model logging encountered an issue: {e}")

    return comparison_df, best_model, best_model_name, best_record


def save_training_outputs(
    best_model: Any,
    best_model_name: str,
    comparison_df: pd.DataFrame,
    best_metrics: Dict[str, Any],
    models_dir: Union[str, Path] = "models",
    reports_dir: Union[str, Path] = "reports",
) -> Dict[str, str]:
    """Saves serialized best model, comparison CSV, and metadata report.

    Args:
        best_model: Fitted best model instance.
        best_model_name: Name of the selected best model.
        comparison_df: Model comparison DataFrame.
        best_metrics: Metrics dictionary of the best model.
        models_dir: Directory for model binaries.
        reports_dir: Directory for comparison reports.

    Returns:
        Dictionary of saved file paths.
    """
    root = get_project_root()
    mod_dir = Path(models_dir) if Path(models_dir).is_absolute() else root / models_dir
    rep_dir = Path(reports_dir) if Path(reports_dir).is_absolute() else root / reports_dir

    mod_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = mod_dir / "best_model.joblib"
    comparison_csv_path = rep_dir / "model_comparison.csv"
    metadata_json_path = rep_dir / "best_model_metadata.json"

    # Save model binary
    joblib.dump(best_model, best_model_path)
    logger.info(f"Saved best model artifact to: {best_model_path}")

    # Save comparison table
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"Saved comparison report to: {comparison_csv_path}")

    # Save metadata JSON
    metadata = {
        "selected_model": best_model_name,
        "selection_criterion": "lowest_cv_rmse",
        "metrics": best_metrics,
        "candidate_models_evaluated": comparison_df["model_name"].tolist(),
    }
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved best model metadata to: {metadata_json_path}")

    return {
        "best_model_path": str(best_model_path),
        "comparison_csv_path": str(comparison_csv_path),
        "metadata_json_path": str(metadata_json_path),
    }


def run_training_pipeline(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Executes the complete Step 5 model training and comparison stage.

    Args:
        config_path: Path to configuration YAML.

    Returns:
        Dictionary containing comparison results and output paths.
    """
    logger.info("================ Starting Stage 05: Model Training & Comparison ================")
    root = get_project_root()
    config = load_yaml(config_path)

    # Resolve paths
    preproc_cfg = config.get("preprocessing", {})
    X_train_path = root / preproc_cfg.get("X_train_path", "data/processed/X_train.csv")
    y_train_path = root / preproc_cfg.get("y_train_path", "data/processed/y_train.csv")

    random_seed = config["project"].get("random_seed", 42)
    models_dir = root / config["models"].get("output_dir", "models")
    reports_dir = root / "reports"

    # MLflow config
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlruns.db")
    experiment_name = mlflow_cfg.get("experiment_name", "Insurance_Prediction")

    logger.info(f"Loading preprocessed training data from: {X_train_path} and {y_train_path}")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    comparison_df, best_model, best_name, best_metrics = train_and_compare_models(
        X_train=X_train,
        y_train=y_train,
        cv_folds=5,
        random_seed=random_seed,
        enable_mlflow=True,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    saved_paths = save_training_outputs(
        best_model=best_model,
        best_model_name=best_name,
        comparison_df=comparison_df,
        best_metrics=best_metrics,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    logger.info("================ Stage 05 Completed Successfully ================")
    return {
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "paths": saved_paths,
    }
