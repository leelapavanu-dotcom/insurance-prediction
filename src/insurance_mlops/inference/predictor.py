"""Inference engine for Insurance Charges Prediction."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Literal

from insurance_mlops.data.validator import validate_schema
from insurance_mlops.utils.common import get_project_root, load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class InsuranceInputRecord(BaseModel):
    """Pydantic schema for inference input validation."""

    age: int = Field(..., ge=18, le=100, description="Age of primary beneficiary (18-100)", examples=[32])
    sex: Literal["female", "male"] = Field(..., description="Gender ('female' or 'male')", examples=["female"])
    bmi: float = Field(..., ge=10.0, le=70.0, description="Body Mass Index (10.0-70.0)", examples=[28.8])
    children: int = Field(..., ge=0, le=20, description="Number of dependents (0-20)", examples=[1])
    smoker: Literal["yes", "no"] = Field(..., description="Smoking status ('yes' or 'no')", examples=["no"])
    region: Literal["northeast", "northwest", "southeast", "southwest"] = Field(
        ..., description="US geographic region", examples=["northwest"]
    )


class SinglePredictionResult(BaseModel):
    """Output schema for a single prediction response."""

    predicted_charges: float
    model_name: str
    model_version: Optional[str] = None


class BatchPredictionResult(BaseModel):
    """Output schema for batch prediction response."""

    predictions: List[float]
    count: int
    model_name: str
    model_version: Optional[str] = None


class InsurancePredictor:
    """Production predictor encapsulating fitted preprocessor and selected regression model."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
        config_path: str = "config/config.yaml",
    ):
        root = get_project_root()
        config = load_yaml(config_path)

        # Resolve paths
        if preprocessor_path is None:
            preproc_rel = config.get("preprocessing", {}).get("preprocessor_path", "models/preprocessor.joblib")
            self.preprocessor_path = root / preproc_rel
        else:
            p = Path(preprocessor_path)
            self.preprocessor_path = p if p.is_absolute() else root / preprocessor_path

        if model_path is None:
            m_dir = config["models"].get("output_dir", "models")
            m_name = config["models"].get("best_model_name", "best_model.joblib")
            self.model_path = root / m_dir / m_name
        else:
            m = Path(model_path)
            self.model_path = m if m.is_absolute() else root / model_path

        # Load artifacts safely
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {self.preprocessor_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")

        logger.info(f"Loading preprocessor from: {self.preprocessor_path}")
        self.preprocessor = joblib.load(self.preprocessor_path)

        logger.info(f"Loading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)

        # Load metadata
        self.model_name = "random_forest"
        self.model_version = "champion"
        lifecycle_file = root / "reports" / "model_lifecycle.json"
        if lifecycle_file.exists():
            try:
                with open(lifecycle_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.model_name = meta.get("architecture", self.model_name)
                    self.model_version = meta.get("alias", meta.get("version", self.model_version))
            except Exception:
                pass

        logger.info(
            f"InsurancePredictor initialized successfully "
            f"(Model: {self.model_name}, Version/Alias: {self.model_version})."
        )

    def predict_records(self, records: List[Dict[str, Any]]) -> List[float]:
        """Validates input records and returns list of predicted insurance charges.

        Args:
            records: List of dictionaries matching the insurance feature schema.

        Returns:
            List of predicted charges rounded to 2 decimal places.

        Raises:
            ValidationError: If any record fails Pydantic schema validation.
        """
        # Validate each record via Pydantic schema
        validated_records: List[Dict[str, Any]] = []
        for r in records:
            validated = InsuranceInputRecord(**r)
            validated_records.append(validated.model_dump())

        df = pd.DataFrame(validated_records)
        feature_names = self.preprocessor.get_feature_names_out()
        df_trans = pd.DataFrame(
            self.preprocessor.transform(df),
            columns=feature_names,
            index=df.index,
        )
        preds = self.model.predict(df_trans)

        return [round(float(p), 2) for p in preds]

    def predict_single(self, record: Dict[str, Any]) -> float:
        """Validates and predicts charges for a single observation.

        Args:
            record: Dictionary of feature attributes.

        Returns:
            Predicted charge amount in USD.
        """
        return self.predict_records([record])[0]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates input DataFrame, computes predictions, and appends 'predicted_charges'.

        Args:
            df: Input pandas DataFrame containing feature columns.

        Returns:
            DataFrame copy containing original features and 'predicted_charges'.
        """
        required_features = ["age", "sex", "bmi", "children", "smoker", "region"]
        missing = [c for c in required_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required features for inference: {missing}")

        # Schema validation (is_training=False allows charges to be absent)
        validated_df, _ = validate_schema(df[required_features], is_training=False, drop_duplicates=False)

        feature_names = self.preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(
            self.preprocessor.transform(validated_df),
            columns=feature_names,
            index=validated_df.index,
        )
        predictions = self.model.predict(transformed_df)

        result_df = df.copy()
        result_df["predicted_charges"] = np.round(predictions, 2)
        return result_df


def predict_batch_csv(
    input_csv_path: Union[str, Path],
    output_csv_path: Union[str, Path],
    predictor: Optional[InsurancePredictor] = None,
) -> Path:
    """Performs batch prediction on an input CSV file and writes results to disk.

    Args:
        input_csv_path: Path to input CSV.
        output_csv_path: Path to destination CSV.
        predictor: Optional pre-instantiated InsurancePredictor.

    Returns:
        Path to output CSV with predictions.
    """
    root = get_project_root()
    in_path = Path(input_csv_path) if Path(input_csv_path).is_absolute() else root / input_csv_path
    out_path = Path(output_csv_path) if Path(output_csv_path).is_absolute() else root / output_csv_path

    if not in_path.exists():
        raise FileNotFoundError(f"Batch input CSV not found at: {in_path}")

    logger.info(f"Loading batch inference data from: {in_path}")
    df_in = pd.read_csv(in_path)
    logger.info(f"Read {len(df_in)} rows for batch prediction.")

    if predictor is None:
        predictor = InsurancePredictor()

    df_out = predictor.predict_dataframe(df_in)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df_out)} predictions to: {out_path}")
    return out_path
