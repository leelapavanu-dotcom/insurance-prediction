# Pipeline Execution Guide — Insurance Cost Prediction MLOps

This guide provides end-to-end instructions for executing every stage of the project on macOS / Linux using Python 3.11.

---

## 1. Environment Activation

```bash
# Navigate to workspace
cd "/Users/pavan/Downloads/Insurance prediction"

# Activate the local virtual environment
source venv/bin/activate
```

---

## 2. Master Pipeline Execution (All Stages in 10s)

To run the complete lifecycle from data ingestion through failure injection in one automated workflow:

```bash
python src/mlops_pipeline/12_end_to_end_pipeline.py
```

**Output**: Generates `reports/end_to_end_execution_summary.json` and prints the stage-by-stage status table.

---

## 3. Individual Stage Execution

You can also run individual modular stages independently:

### Stage 01/02: Ingestion & Validation
```bash
python -c 'from insurance_mlops.data.loader import run_data_pipeline; run_data_pipeline()'
```
- Validates raw data against Pydantic schema.
- Splits into `data/processed/train.csv` (1,070 rows) and `test.csv` (268 rows).
- Saves `data/reference/reference.csv` baseline.

### Stage 03: Feature Preprocessing
```bash
python src/mlops_pipeline/03_preprocessing.py
```
- Fits `ColumnTransformer` on training data only.
- Saves `models/preprocessor.joblib`.
- Generates `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.

### Stage 04: Validate Preprocessing Outputs
```bash
python src/mlops_pipeline/04_validate_outputs.py
```
- Performs integrity checks on generated matrices (zero NaNs, standardized features).

### Stage 05: Model Training, Comparison & MLflow Logging
```bash
python src/mlops_pipeline/05_train_registry.py
```
- Evaluates Linear Regression, Random Forest, and Gradient Boosting with 5-Fold CV.
- Logs parameters, metrics, and models to MLflow (`sqlite:///mlruns.db`).
- Saves `models/best_model.joblib` and `reports/model_comparison.csv`.

### Stage 06: Held-Out Model Evaluation
```bash
python src/mlops_pipeline/06_model_evaluation.py
```
- Evaluates winning model strictly on 268 unseen held-out test rows.
- Saves `reports/test_evaluation.json` and `reports/test_evaluation.csv`.

### Stage 07: Quality Gate Enforcement
```bash
python src/mlops_pipeline/07_quality_gate.py
```
- Verifies model meets $R^2 \ge 0.80$, $\text{RMSE} \le \$5,200$, $\text{MAE} \le \$3,000$, $\Delta R^2 \le 0.15$.
- Saves decision to `reports/quality_gate.json`.

### Stage 08: Model Registry & Champion Alias Promotion
```bash
python src/mlops_pipeline/08_automate_lifecycle.py
```
- Promotes model in MLflow Model Registry and assigns the `@champion` alias.
- Saves state to `reports/model_lifecycle.json`.

### Stage 09: Batch Production Inference
```bash
python src/mlops_pipeline/09_inference.py \
  --input-path data/processed/sample_inference_input.csv \
  --output-path data/processed/batch_predictions.csv
```
- Runs vectorized batch predictions and writes results.

### Stage 10: Statistical Data Drift Monitoring
```bash
python src/mlops_pipeline/10_drift_monitoring.py \
  --input-path data/processed/test.csv
```
- Compares distributions using KS-tests (numerical) and Chi-Square tests (categorical).
- Outputs `reports/drift_report.json` and `reports/drift_report.csv`.

### Stage 11: Resilience & Failure Injection
```bash
python src/mlops_pipeline/11_failure_injection.py
```
- Tests 10 safe failure scenarios and outputs `reports/failure_injection_report.json`.

---

## 4. MLflow UI Dashboard

Launch the local MLflow Tracking server to inspect experiments and model versions:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```
Open [http://localhost:5000](http://localhost:5000) in your browser:
- **Experiments**: View run comparisons, RMSE, MAE, R², and logged artifact files.
- **Models**: View registered model `insurance_charges_regressor` and the `@champion` alias.

---

## 5. FastAPI Service Execution

Launch the inference microservice locally:

```bash
uvicorn insurance_mlops.inference.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Test Endpoints:
```bash
# Health probe
curl -X GET http://localhost:8000/health

# Prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "sex": "male",
    "bmi": 27.9,
    "children": 1,
    "smoker": "no",
    "region": "northwest"
  }'
```

---

## 6. Testing & Quality Assurance

```bash
# Run full automated test suite (71 tests)
pytest tests/ -v

# Run with test coverage report
pytest tests/ -v --cov=src/insurance_mlops --cov-report=term-missing

# Run code style and PEP 8 linting
flake8 src tests
```

---

## 7. Docker & Kubernetes Execution

### Docker Build & Run:
```bash
# Build production image
docker build -t insurance-mlops-api:latest .

# Run container
docker run -d --name insurance-api -p 8000:8000 insurance-mlops-api:latest

# Check health
curl http://localhost:8000/health
```

### Docker Compose:
```bash
docker compose up -d
docker compose ps
docker compose down
```

### Kubernetes (Minikube / Docker Desktop):
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/insurance-api-deployment
```
