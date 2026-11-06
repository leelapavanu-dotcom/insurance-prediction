"""Production monitoring service and drift report generator."""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import pandas as pd

from insurance_mlops.monitoring.drift_detector import DriftDetector, FeatureDriftResult
from insurance_mlops.utils.common import get_project_root, load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class DataDriftMonitor:
    """Production monitor managing reference baseline comparison and drift reporting."""

    def __init__(self, config_path: str = "config/config.yaml"):
        root = get_project_root()
        self.config = load_yaml(config_path)

        mon_cfg = self.config.get("monitoring", {})
        self.p_value_threshold = float(mon_cfg.get("p_value_threshold", 0.05))
        self.alert_drift_share_threshold = float(mon_cfg.get("alert_drift_share_threshold", 0.33))

        ref_rel = mon_cfg.get("reference_path", "data/reference/reference.csv")
        self.reference_path = root / ref_rel

        rep_json_rel = mon_cfg.get("report_json_path", "reports/drift_report.json")
        rep_csv_rel = mon_cfg.get("report_csv_path", "reports/drift_report.csv")
        self.report_json_path = root / rep_json_rel
        self.report_csv_path = root / rep_csv_rel

        num_features = self.config["features"]["numerical"]
        cat_features = self.config["features"]["categorical"]

        self.detector = DriftDetector(
            p_value_threshold=self.p_value_threshold,
            numerical_features=num_features,
            categorical_features=cat_features,
        )
        logger.info(
            f"DataDriftMonitor initialized (p_threshold={self.p_value_threshold}, "
            f"alert_share_threshold={self.alert_drift_share_threshold})."
        )

    def load_reference_data(self) -> pd.DataFrame:
        """Loads baseline reference dataset without modifying it."""
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference dataset not found at: {self.reference_path}")
        logger.info(f"Loading baseline reference data from: {self.reference_path}")
        df = pd.read_csv(self.reference_path)
        logger.info(f"Loaded reference dataset: {len(df)} rows.")
        return df

    def run_monitoring(
        self,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Compares incoming production data against the reference baseline and outputs reports.

        Args:
            current_df: Current incoming production or test DataFrame.
            reference_df: Optional reference DataFrame. If None, loads from reference_path.

        Returns:
            Dictionary containing drift summary, alert status, and feature results.
        """
        logger.info(f"Starting data drift evaluation against {len(current_df)} current observations...")
        if reference_df is None:
            reference_df = self.load_reference_data()

        # Run drift tests
        feature_results: List[FeatureDriftResult] = self.detector.detect_dataset_drift(
            reference_df=reference_df,
            current_df=current_df,
        )

        total_features = len(feature_results)
        drifted_features = [r.feature_name for r in feature_results if r.drift_detected]
        drift_count = len(drifted_features)
        drift_share = round(float(drift_count / total_features) if total_features > 0 else 0.0, 4)

        alert_triggered = bool(drift_share >= self.alert_drift_share_threshold)
        status = "DRIFT_ALERT" if alert_triggered else "HEALTHY"

        if alert_triggered:
            logger.warning(
                f"🚨 DRIFT ALERT TRIGGERED: {drift_count}/{total_features} features drifted "
                f"({drift_share:.1%} >= {self.alert_drift_share_threshold:.1%}). Drifted: {drifted_features}"
            )
        else:
            logger.info(
                f"✓ DRIFT STATUS HEALTHY: {drift_count}/{total_features} features drifted ({drift_share:.1%})."
            )

        # Convert to serializable format
        results_data = [r.model_dump() for r in feature_results]

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "alert_triggered": alert_triggered,
            "alert_threshold_share": self.alert_drift_share_threshold,
            "total_features_evaluated": total_features,
            "drifted_features_count": drift_count,
            "drift_share": drift_share,
            "drifted_features": drifted_features,
            "reference_sample_size": len(reference_df),
            "current_sample_size": len(current_df),
            "feature_drift_results": results_data,
        }

        # Save to JSON
        self.report_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved drift JSON report to: {self.report_json_path}")

        # Save to CSV
        df_csv = pd.DataFrame(results_data)
        df_csv.to_csv(self.report_csv_path, index=False)
        logger.info(f"Saved drift CSV report to: {self.report_csv_path}")

        return summary
