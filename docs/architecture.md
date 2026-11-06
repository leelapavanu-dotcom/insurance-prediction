# System Architecture — Insurance Cost Prediction MLOps

## 1. High-Level Architectural Overview

The **Insurance Cost Prediction MLOps System** is designed around modular, production-grade principles to estimate individual healthcare charges based on demographic and lifestyle indicators. The system emphasizes:
- **Zero Data Leakage Boundary**: Preprocessing transformers are fit strictly on training partitions and serialized.
- **Contract-Driven Validation**: Schema and domain constraints are enforced using Pydantic.
- **Automated Quality Gates**: Objective performance criteria prevent substandard models from reaching production.
- **Enterprise Lifecycle Management**: Model versioning and alias promotion (`@champion`) via MLflow Model Registry.
- **Production Serving**: Dual inference paths via Batch CLI and a FastAPI REST service.
- **Continuous Monitoring**: Two-sample Kolmogorov-Smirnov and Chi-Square statistical drift tests.
- **Cloud-Native Deployment**: Non-root Docker containerization and Kubernetes manifests.

---

## 2. End-to-End Pipeline Architecture (Mermaid)

```mermaid
flowchart TD
    subgraph Data_Layer ["Data Ingestion & Validation"]
        RawCSV["data/raw/insurance.csv<br/>(1,338 records)"] --> Ingestion["Stage 01/02: Ingestion & Pydantic Validation"]
        Ingestion --> TrainCSV["data/processed/train.csv<br/>(1,070 rows, 80%)"]
        Ingestion --> TestCSV["data/processed/test.csv<br/>(268 rows, 20%)"]
        Ingestion --> RefCSV["data/reference/reference.csv<br/>(Baseline Distribution)"]
    end

    subgraph Preproc_Layer ["Feature Engineering & Preprocessing"]
        TrainCSV --> PreprocFit["Stage 03: Fit ColumnTransformer<br/>(StandardScaler + OneHotEncoder)"]
        PreprocFit --> PrepArtifact["models/preprocessor.joblib"]
        PrepArtifact --> TransformTrain["Transform Train"] --> XTrain["data/processed/X_train.csv"]
        PrepArtifact --> TransformTest["Transform Test"] --> XTest["data/processed/X_test.csv"]
        TransformTrain --> ValPrep["Stage 04: Validate Preprocessing Outputs"]
    end

    subgraph Model_Layer ["Training & Experiment Tracking"]
        XTrain --> TrainComp["Stage 05: 5-Fold Cross Validation<br/>(Linear, Random Forest, Gradient Boosting)"]
        TrainComp --> MLflowExp["MLflow Tracking (sqlite:///mlruns.db)"]
        TrainComp --> BestModel["models/best_model.joblib<br/>(RandomForestRegressor)"]
        TrainComp --> CompCSV["reports/model_comparison.csv"]
    end

    subgraph Eval_Gate_Layer ["Evaluation, Quality Gates & Registry"]
        XTest --> HeldOutEval["Stage 06: Held-Out Evaluation<br/>(268 unseen records)"]
        BestModel --> HeldOutEval
        HeldOutEval --> EvalJSON["reports/test_evaluation.json"]
        EvalJSON --> QualityGate["Stage 07: Quality Gate Verification<br/>(R² >= 0.80, RMSE <= $5,200, Overfitting <= 0.15)"]
        QualityGate -->|PASS| RegPromote["Stage 08: Lifecycle & Alias Promotion"]
        RegPromote --> ChampAlias["MLflow Registry (@champion alias)"]
        QualityGate -->|FAIL| Reject["Model Promotion Blocked"]
    end

    subgraph Serving_Layer ["Inference & Production Serving"]
        ChampAlias --> Predictor["InsurancePredictor Engine"]
        PrepArtifact --> Predictor
        BestModel --> Predictor
        Predictor --> BatchCLI["Stage 09: Batch Prediction CLI"]
        Predictor --> FastAPISvc["FastAPI REST Service (/predict)"]
    end

    subgraph Observability_Layer ["Observability & Resilience"]
        RefCSV --> DriftMon["Stage 10: Statistical Drift Monitor<br/>(KS-Test & Chi-Square Tests)"]
        FastAPISvc -. Incoming Data .-> DriftMon
        DriftMon --> DriftReport["reports/drift_report.json"]
        DriftMon -->|Drift Share >= 33%| Alert["🚨 Production Drift Alert"]
        DriftMon -->|Drift Share < 33%| Healthy["✓ Distribution Healthy"]
        
        FailInj["Stage 11: Failure Injection & Resilience<br/>(10 Safe Stress Scenarios)"] --> ResilienceReport["reports/failure_injection_report.json"]
    end
```

---

## 3. Core Subsystems

### A. Data Ingestion & Validation
- **Pydantic Data Contracts**: Validates demographic ranges (`age: 18-100`, `bmi: 10.0-70.0`, `children: 0-20`) and categorical enumerations (`sex`, `smoker`, `region`).
- **Configurable Deduplication**: Reports duplicate records (e.g. index 581) and preserves or drops them based on configuration (`drop_duplicates: false`).

### B. Feature Preprocessing Pipeline
- **Numerical Pipeline**: Continuous variables (`age`, `bmi`, `children`) transformed via `StandardScaler`.
- **Categorical Pipeline**: Nominal features (`sex`, `smoker`, `region`) encoded via `OneHotEncoder(handle_unknown='ignore')`.
- **Integrity Validation**: Verifies zero NaNs, infinite values, and row-count alignment between features and targets.

### C. Model Training & Comparison
- Evaluates 3 distinct model architectures using 5-Fold Cross Validation:
  1. `linear_regression` (Baseline)
  2. `random_forest` (Selected Best Model)
  3. `gradient_boosting` (Competitive Ensemble)
- Winning model artifact serialized to `models/best_model.joblib`.

### D. Quality Gates & Lifecycle Management
- Requires models to exceed stringent criteria before promotion:
  - Held-out $R^2 \ge 0.80$
  - Held-out $\text{RMSE} \le \$5,200$
  - Held-out $\text{MAE} \le \$3,000$
  - Overfitting Delta $\Delta R^2 \le 0.15$
- Assigns `@champion` alias in the MLflow Model Registry.

### E. Inference & Containerized Serving
- **Batch Pipeline**: Efficient vectorized prediction on CSV datasets.
- **FastAPI Microservice**: Exposes `GET /health` and `POST /predict` supporting both single observations and high-throughput batches.
- **Docker Container**: Non-root runtime with health checks.
- **Kubernetes**: Scalable deployment with rolling updates, resource quotas, and ClusterIP service routing.

### F. Statistical Drift Monitoring
- **Two-Sample Kolmogorov-Smirnov (KS) Test**: Detects continuous feature distribution shifts.
- **Chi-Square Contingency Test**: Detects categorical category frequency shifts.
- **Dynamic Alerting**: Triggers production alerts when drifted feature share exceeds 33%.
