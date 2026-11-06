# Troubleshooting Guide — Insurance Cost Prediction MLOps

This guide documents common operational issues, error diagnostics, and verified solutions.

---

## 1. Missing Model or Preprocessor Artifacts

### Symptoms:
- `FileNotFoundError: Preprocessor artifact not found at: models/preprocessor.joblib`
- `FileNotFoundError: Best model artifact not found at: models/best_model.joblib`
- API returns `HTTP 503 Service Unavailable: Model artifacts not available`.

### Root Cause:
The preprocessing or training stages have not yet been executed in the current environment, or files were cleaned up.

### Solution:
Execute the pipeline stages to regenerate the artifacts:
```bash
python src/mlops_pipeline/03_preprocessing.py
python src/mlops_pipeline/05_train_registry.py
```
Or execute the master pipeline:
```bash
python src/mlops_pipeline/12_end_to_end_pipeline.py
```

---

## 2. Quality Gate Rejection (`REJECTED_QUALITY_GATE_FAILED`)

### Symptoms:
- `stage_08_automate_lifecycle.py` halts with `Model promotion BLOCKED: Quality gate failed`.
- `reports/quality_gate.json` contains `"status": "FAIL"`.

### Root Cause:
The trained candidate model failed to meet one or more criteria in `config/quality_gates.yaml`:
- $R^2 < 0.80$
- $\text{RMSE} > \$5,200$
- $\text{MAE} > \$3,000$
- Overfitting Delta $\Delta R^2 > 0.15$

### Solution:
1. Inspect `reports/test_evaluation.json` and `reports/quality_gate.json` to identify which metric failed.
2. If data volume or noise shifted legitimately, adjust thresholds in `config/quality_gates.yaml` after team review.
3. If the model underperformed, tune hyperparameters in `src/insurance_mlops/models/trainer.py` (e.g. increase `n_estimators` or tune `max_depth`).

---

## 3. SQLite Database Lock (`database is locked`)

### Symptoms:
- MLflow logs error: `sqlite3.OperationalError: database is locked`.

### Root Cause:
Multiple concurrent processes attempting to write simultaneously to `sqlite:///mlruns.db`.

### Solution:
1. Ensure no background MLflow runs are hung:
   ```bash
   lsof mlruns.db
   ```
2. Kill any stale process accessing the database.
3. For heavy concurrent multi-agent training, switch `tracking_uri` in `config/config.yaml` to PostgreSQL or an MLflow Tracking Server.

---

## 4. Port 8000 Already in Use

### Symptoms:
- `uvicorn` or `docker run` fails with `[Errno 48] Address already in use`.

### Root Cause:
An existing FastAPI server or background process is already bound to port 8000.

### Solution:
Find and terminate the process:
```bash
lsof -i :8000
kill -9 <PID>
```
Or bind to a different port:
```bash
uvicorn insurance_mlops.inference.api:app --host 0.0.0.0 --port 8080
```

---

## 5. Data Drift Alert Triggered (`DRIFT_ALERT`)

### Symptoms:
- `10_drift_monitoring.py` reports `🚨 DRIFT ALERT TRIGGERED: X/6 features drifted`.

### Root Cause:
The distribution of incoming production data deviates significantly ($p < 0.05$) from the baseline in `data/reference/reference.csv` across $\ge 33\%$ of features.

### Solution:
1. Open `reports/drift_report.csv` to see which features drifted and their statistical p-values.
2. Determine if the drift represents a seasonal change, population demographic shift, or upstream data corruption.
3. If genuine demographic change has occurred:
   - Collect fresh production labels.
   - Update `data/reference/reference.csv` with recent data.
   - Retrain the champion model via `python src/mlops_pipeline/12_end_to_end_pipeline.py`.

---

## 6. Docker Daemon Connection Issues

### Symptoms:
- `failed to connect to the docker API at unix:///var/run/docker.sock`.

### Root Cause:
Docker Desktop is not running, or the executing environment lacks permissions to the Unix socket.

### Solution:
1. Start the Docker Desktop application on macOS.
2. Verify daemon status:
   ```bash
   docker info
   ```
3. If running inside a restricted sandbox, build the Docker image in an elevated terminal session.

---

## 7. Kubernetes Pod Readiness / Liveness Probe Failures

### Symptoms:
- `kubectl get pods` shows `0/1 Running` or `CrashLoopBackOff`.
- `Readiness probe failed: HTTP probe failed with statuscode: 503`.

### Root Cause:
The container is healthy, but the model artifact is missing from the image context.

### Solution:
1. Ensure `models/best_model.joblib` and `models/preprocessor.joblib` exist before building the Docker image.
2. Rebuild the image:
   ```bash
   docker build -t insurance-mlops-api:latest .
   ```
3. If using Minikube, reload the image:
   ```bash
   minikube image load insurance-mlops-api:latest
   ```
4. Restart the deployment:
   ```bash
   kubectl rollout restart deployment/insurance-api-deployment
   ```
