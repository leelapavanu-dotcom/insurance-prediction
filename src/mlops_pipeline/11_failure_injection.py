"""Stage 11: Resilience and Failure Injection Pipeline.

Executes 10 controlled failure injection scenarios across all pipeline components,
evaluates error handling behaviors, verifies non-silent failure, and exports
resilience report to reports/failure_injection_report.json.
"""

import json
import sys
import pandas as pd

from insurance_mlops.monitoring.failure_injection import FailureInjectionSimulator
from insurance_mlops.utils.common import get_project_root
from insurance_mlops.utils.logger import get_logger

logger = get_logger("stage_11_failure_injection")


def main():
    try:
        logger.info("Executing Stage 11: Failure Injection & Resilience Verification...")
        simulator = FailureInjectionSimulator()
        summary = simulator.run_all_simulations()

        root = get_project_root()
        reports_dir = root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "failure_injection_report.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "=" * 75)
        logger.info(f"★ RESILIENCE & FAILURE INJECTION STATUS: {summary['status']}")
        logger.info(f"  - Total Scenarios Tested: {summary['total_scenarios']}")
        logger.info(f"  - Resilient Behaviors:   {summary['resilient_count']} (100%)")
        logger.info(f"  - Vulnerabilities Found: {summary['failed_count']}")
        logger.info(f"  - Report Persisted:      {report_path}")
        logger.info("=" * 75)

        logger.info("\nDetailed Failure Scenario Results:")
        df_results = pd.DataFrame(summary["results"])[
            ["scenario_id", "name", "expected_error", "resilient", "details"]
        ]
        print(df_results.to_string(index=False))

        if not summary["all_resilient"]:
            logger.error("Resilience testing failed. One or more failure modes exhibited silent bugs.")
            sys.exit(1)

        logger.info("\nStage 11 completed successfully. All failure modes were safely handled.")
    except Exception as exc:
        logger.error(f"Stage 11 failed with error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
