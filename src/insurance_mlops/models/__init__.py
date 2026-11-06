"""Model training, evaluation, and registry module."""

from insurance_mlops.models.registry import register_and_promote_model
from insurance_mlops.models.trainer import (
    calculate_metrics,
    cross_validate_model,
    get_model_candidates,
    run_training_pipeline,
    save_training_outputs,
    train_and_compare_models,
)

__all__ = [
    "calculate_metrics",
    "get_model_candidates",
    "cross_validate_model",
    "train_and_compare_models",
    "save_training_outputs",
    "run_training_pipeline",
    "register_and_promote_model",
]
