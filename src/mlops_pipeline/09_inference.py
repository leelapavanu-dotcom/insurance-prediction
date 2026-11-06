"""Stage 09: Batch Prediction and Offline Inference Pipeline.

Reads a batch of unseen feature observations, executes schema validation,
applies the serialized preprocessor, computes predictions with the champion model,
and outputs a CSV with 'predicted_charges'.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

from insurance_mlops.inference.predictor import InsurancePredictor, predict_batch_csv
from insurance_mlops.utils.common import get_project_root, load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger("stage_09_inference")


def create_sample_inference_file_if_missing(sample_path: Path) -> Path:
    """Creates a sample inference CSV if one does not already exist."""
    if not sample_path.exists():
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_data = pd.DataFrame([
            {"age": 19, "sex": "female", "bmi": 27.90, "children": 0, "smoker": "yes", "region": "southwest"},
            {"age": 18, "sex": "male", "bmi": 33.77, "children": 1, "smoker": "no", "region": "southeast"},
            {"age": 28, "sex": "male", "bmi": 33.00, "children": 3, "smoker": "no", "region": "southeast"},
            {"age": 33, "sex": "male", "bmi": 22.70, "children": 0, "smoker": "no", "region": "northwest"},
            {"age": 54, "sex": "female", "bmi": 47.41, "children": 0, "smoker": "yes", "region": "southeast"},
        ])
        sample_data.to_csv(sample_path, index=False)
        logger.info(f"Created sample inference input CSV at: {sample_path} (5 records)")
    return sample_path


def main():
    parser = argparse.ArgumentParser(description="Run Batch Inference for Insurance Charges")
    parser.add_argument("--input", type=str, default=None, help="Input CSV file path")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Configuration file path")
    args = parser.parse_args()

    try:
        logger.info("Executing Stage 09: Batch Inference Pipeline...")
        root = get_project_root()
        config = load_yaml(args.config)

        # Resolve paths
        inf_cfg = config.get("inference", {})
        default_in = root / inf_cfg.get("sample_input_path", "data/processed/sample_inference_input.csv")
        default_out = root / inf_cfg.get("batch_output_path", "data/processed/batch_predictions.csv")

        in_path = Path(args.input) if args.input else default_in
        out_path = Path(args.output) if args.output else default_out

        # Ensure sample input exists if using default
        if not in_path.exists():
            create_sample_inference_file_if_missing(in_path)

        predictor = InsurancePredictor(config_path=args.config)
        saved_file = predict_batch_csv(
            input_csv_path=in_path,
            output_csv_path=out_path,
            predictor=predictor,
        )

        # Display preview of results
        df_result = pd.read_csv(saved_file)
        logger.info("\n" + "=" * 65)
        logger.info("★ BATCH INFERENCE PREVIEW (First 5 Rows):")
        logger.info("=" * 65)
        print(df_result.to_string(index=False))
        logger.info("=" * 65)
        logger.info(f"Batch prediction completed successfully for {len(df_result)} records -> {saved_file}")

    except Exception as exc:
        logger.error(f"Stage 09 Batch Inference failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
