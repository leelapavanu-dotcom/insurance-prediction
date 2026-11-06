"""Stage 10: Production Data Drift Monitoring Pipeline.

Compares incoming production data against the baseline reference dataset,
executes Kolmogorov-Smirnov and Chi-Square statistical tests across all features,
and generates structured JSON & CSV drift audit reports.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

from insurance_mlops.monitoring.monitor import DataDriftMonitor
from insurance_mlops.utils.common import get_project_root, load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger("stage_10_drift_monitoring")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Data Drift against Reference Baseline")
    parser.add_argument("--input", type=str, default=None, help="Path to current data CSV (defaults to test.csv)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Configuration file path")
    args = parser.parse_args()

    try:
        logger.info("Executing Stage 10: Data Drift Monitoring Pipeline...")
        root = get_project_root()
        config = load_yaml(args.config)

        # Default current data to held-out test set
        default_test = root / config["data"].get("test_path", "data/processed/test.csv")
        current_csv = Path(args.input) if args.input else default_test

        if not current_csv.exists():
            raise FileNotFoundError(f"Target dataset for drift monitoring not found at: {current_csv}")

        logger.info(f"Loading target dataset from: {current_csv}")
        current_df = pd.read_csv(current_csv)

        monitor = DataDriftMonitor(config_path=args.config)
        summary = monitor.run_monitoring(current_df=current_df)

        logger.info("\n" + "=" * 70)
        logger.info(f"★ DATA DRIFT MONITORING STATUS: {summary['status']}")
        logger.info(f"  - Features Evaluated: {summary['total_features_evaluated']}")
        logger.info(f"  - Drifted Features:   {summary['drifted_features_count']} ({summary['drift_share']:.1%})")
        logger.info(f"  - Alert Threshold:    {summary['alert_threshold_share']:.1%}")
        logger.info(f"  - Alert Triggered:    {summary['alert_triggered']}")
        logger.info("=" * 70)

        logger.info("\nFeature-Level Statistical Test Results:")
        df_results = pd.DataFrame(summary["feature_drift_results"])
        display_cols = [
            "feature_name", "feature_type", "test_name",
            "statistic", "p_value", "drift_detected", "severity"
        ]
        print(df_results[display_cols].to_string(index=False))

        logger.info("\nDrift Reports Saved:")
        logger.info(f"  - JSON: {monitor.report_json_path}")
        logger.info(f"  - CSV:  {monitor.report_csv_path}")
        logger.info("Stage 10 completed successfully.")

    except Exception as exc:
        logger.error(f"Stage 10 Data Drift Monitoring failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
