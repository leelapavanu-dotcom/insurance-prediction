"""Model evaluation on held-out test data and Quality Gate enforcement."""

import json
from typing import Any, Dict, List, Optional, Union
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from insurance_mlops.utils.common import get_project_root, load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Computes regression metrics for given true and predicted target values.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary with 'rmse', 'mae', and 'r2'.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }


def evaluate_held_out_test(
    config_path: str = "config/config.yaml",
    enable_mlflow: bool = True,
) -> Dict[str, Any]:
    """Evaluates the serialized best model strictly on the held-out test dataset.

    Saves evaluation reports to reports/test_evaluation.json and reports/test_evaluation.csv,
    and logs evaluation metrics to MLflow.

    Args:
        config_path: Path to configuration YAML.
        enable_mlflow: Whether to log results to MLflow.

    Returns:
        Dictionary containing evaluation metrics and file paths.
    """
    logger.info("================ Starting Held-Out Model Evaluation ================")
    root = get_project_root()
    config = load_yaml(config_path)

    # Resolve paths
    preproc_cfg = config.get("preprocessing", {})
    X_test_path = root / preproc_cfg.get("X_test_path", "data/processed/X_test.csv")
    y_test_path = root / preproc_cfg.get("y_test_path", "data/processed/y_test.csv")
    model_dir = config["models"].get("output_dir", "models")
    best_model_name = config["models"].get("best_model_name", "best_model.joblib")
    model_path = root / model_dir / best_model_name
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Best model artifact not found at: {model_path}")
    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(f"Held-out test data not found at {X_test_path} or {y_test_path}")

    # Load artifacts and test data
    logger.info(f"Loading best model artifact from: {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading held-out test data from: {X_test_path}")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).to_numpy().ravel()

    # Predict & Evaluate
    logger.info(f"Generating predictions on {len(X_test)} held-out test observations...")
    y_pred = model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_pred)

    logger.info(
        f"Held-Out Evaluation Results: RMSE=${test_metrics['rmse']:,.2f}, "
        f"MAE=${test_metrics['mae']:,.2f}, R2={test_metrics['r2']:.4f}"
    )

    # Load train metadata if available to compute overfitting delta
    train_metadata_path = reports_dir / "best_model_metadata.json"
    train_r2 = None
    delta_r2 = None
    selected_model_name = "unknown"
    if train_metadata_path.exists():
        with open(train_metadata_path, "r", encoding="utf-8") as f:
            t_meta = json.load(f)
            selected_model_name = t_meta.get("selected_model", "unknown")
            train_r2 = t_meta.get("metrics", {}).get("train_r2")
            if train_r2 is not None:
                delta_r2 = round(float(train_r2 - test_metrics["r2"]), 4)
                logger.info(f"Overfitting Delta R2 (Train R2 - Test R2): {delta_r2:.4f}")

    # Save to reports/test_evaluation.json
    json_path = reports_dir / "test_evaluation.json"
    eval_report = {
        "model_name": selected_model_name,
        "metrics": test_metrics,
        "train_r2": train_r2,
        "delta_r2": delta_r2,
        "test_sample_count": len(X_test),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"Saved test evaluation report to: {json_path}")

    # Save to reports/test_evaluation.csv
    csv_path = reports_dir / "test_evaluation.csv"
    pd.DataFrame([eval_report["metrics"]]).to_csv(csv_path, index=False)
    logger.info(f"Saved test evaluation table to: {csv_path}")

    # Log to MLflow
    if enable_mlflow:
        mlflow_cfg = config.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlruns.db")
        experiment_name = mlflow_cfg.get("experiment_name", "Insurance_Prediction")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        try:
            with mlflow.start_run(run_name="eval_heldout_test") as run:
                mlflow.log_param("evaluation_dataset", "held_out_test_set")
                mlflow.log_param("test_sample_count", len(X_test))
                mlflow.log_metric("test_rmse", test_metrics["rmse"])
                mlflow.log_metric("test_mae", test_metrics["mae"])
                mlflow.log_metric("test_r2", test_metrics["r2"])
                if delta_r2 is not None:
                    mlflow.log_metric("overfitting_delta_r2", delta_r2)
                mlflow.set_tag("stage", "held_out_evaluation")
                logger.info(f"Logged held-out evaluation to MLflow run_id: {run.info.run_id}")
        except Exception as exc:
            logger.warning(f"MLflow evaluation logging issue: {exc}")

    logger.info("================ Held-Out Evaluation Completed Successfully ================")
    return {
        "metrics": test_metrics,
        "delta_r2": delta_r2,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }


def check_quality_gates(
    test_metrics: Dict[str, float],
    train_r2: Optional[float] = None,
    quality_gates_path: str = "config/quality_gates.yaml",
) -> Dict[str, Any]:
    """Validates model evaluation metrics against configured quality gates.

    Args:
        test_metrics: Dictionary containing 'rmse', 'mae', 'r2'.
        train_r2: Optional train R2 score to check overfitting delta.
        quality_gates_path: Path to quality gates YAML configuration.

    Returns:
        Dictionary containing gate result (passed: bool), detailed checks, and failure reasons.
    """
    logger.info("================ Evaluating Model Against Quality Gates ================")
    root = get_project_root()
    q_config = load_yaml(quality_gates_path)

    q_rules = q_config.get("quality_gates", {})
    if not q_rules.get("enabled", True):
        logger.warning("Quality gates are currently disabled in configuration.")
        return {"passed": True, "status": "disabled", "checks": []}

    metrics_cfg = q_rules.get("metrics", {})
    promo_rules = q_rules.get("promotion_rules", {})

    checks: List[Dict[str, Any]] = []
    failure_reasons: List[str] = []
    all_passed = True

    # 1. R2 Gate (Higher is better)
    r2_threshold = metrics_cfg.get("r2_score", {}).get("min_threshold")
    actual_r2 = test_metrics.get("r2", 0.0)
    if r2_threshold is not None:
        passed = actual_r2 >= r2_threshold
        if not passed:
            all_passed = False
            msg = f"R² Score check FAILED: Actual={actual_r2:.4f} < Threshold={r2_threshold:.4f}"
            failure_reasons.append(msg)
            logger.error(msg)
        else:
            logger.info(f"✓ R² Score PASSED: Actual={actual_r2:.4f} >= Threshold={r2_threshold:.4f}")
        checks.append({
            "metric": "r2_score",
            "actual": actual_r2,
            "threshold": r2_threshold,
            "condition": ">=",
            "passed": passed,
        })

    # 2. RMSE Gate (Lower is better)
    rmse_threshold = metrics_cfg.get("rmse", {}).get("max_threshold")
    actual_rmse = test_metrics.get("rmse", float("inf"))
    if rmse_threshold is not None:
        passed = actual_rmse <= rmse_threshold
        if not passed:
            all_passed = False
            msg = f"RMSE check FAILED: Actual=${actual_rmse:,.2f} > Threshold=${rmse_threshold:,.2f}"
            failure_reasons.append(msg)
            logger.error(msg)
        else:
            logger.info(f"✓ RMSE PASSED: Actual=${actual_rmse:,.2f} <= Threshold=${rmse_threshold:,.2f}")
        checks.append({
            "metric": "rmse",
            "actual": actual_rmse,
            "threshold": rmse_threshold,
            "condition": "<=",
            "passed": passed,
        })

    # 3. MAE Gate (Lower is better)
    mae_threshold = metrics_cfg.get("mae", {}).get("max_threshold")
    actual_mae = test_metrics.get("mae", float("inf"))
    if mae_threshold is not None:
        passed = actual_mae <= mae_threshold
        if not passed:
            all_passed = False
            msg = f"MAE check FAILED: Actual=${actual_mae:,.2f} > Threshold=${mae_threshold:,.2f}"
            failure_reasons.append(msg)
            logger.error(msg)
        else:
            logger.info(f"✓ MAE PASSED: Actual=${actual_mae:,.2f} <= Threshold=${mae_threshold:,.2f}")
        checks.append({
            "metric": "mae",
            "actual": actual_mae,
            "threshold": mae_threshold,
            "condition": "<=",
            "passed": passed,
        })

    # 4. Overfitting Delta R2 Gate
    delta_threshold = promo_rules.get("max_overfitting_delta_r2")
    if delta_threshold is not None and train_r2 is not None:
        delta_r2 = round(float(train_r2 - actual_r2), 4)
        passed = delta_r2 <= delta_threshold
        if not passed:
            all_passed = False
            msg = f"Overfitting check FAILED: Delta R² ({delta_r2:.4f}) > Max Allowed ({delta_threshold:.4f})"
            failure_reasons.append(msg)
            logger.error(msg)
        else:
            logger.info(f"✓ Overfitting Guardrail PASSED: Delta R² ({delta_r2:.4f}) <= Allowed ({delta_threshold:.4f})")
        checks.append({
            "metric": "overfitting_delta_r2",
            "actual": delta_r2,
            "threshold": delta_threshold,
            "condition": "<=",
            "passed": passed,
        })

    # Save to reports/quality_gate.json
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "quality_gate.json"

    result_payload = {
        "passed": all_passed,
        "status": "PASS" if all_passed else "FAIL",
        "failure_reasons": failure_reasons,
        "checks": checks,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    logger.info(f"Saved quality gate report to: {report_path}")

    if all_passed:
        logger.info("================ Quality Gate Status: ALL CHECKS PASSED ================")
    else:
        logger.error(
            f"================ Quality Gate Status: FAILED ({len(failure_reasons)} violations) ================"
        )

    return result_payload
