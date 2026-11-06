# Project Summary — Insurance Cost Prediction MLOps

## 1. Executive Summary

The **Insurance Cost Prediction MLOps Project** establishes an enterprise-grade, reproducible machine learning operations pipeline for predicting healthcare charges billed to individuals. Developed for **macOS** with **Python 3.11**, the system addresses the full machine learning lifecycle from data validation through Kubernetes-ready deployment.

---

## 2. Dataset & Problem Formulation

- **Dataset**: US Medical Insurance Demographics (`data/raw/insurance.csv`).
- **Observations**: 1,338 records, 7 attributes.
- **Target Variable**: Continuous numerical individual charges (`charges` in USD).
- **Problem Type**: Supervised Regression.
- **Feature Set**:
  - `age` (integer, 18–64)
  - `sex` (`female`, `male`)
  - `bmi` (float, 15.96–53.13)
  - `children` (integer, 0–5)
  - `smoker` (`yes`, `no`)
  - `region` (`northeast`, `northwest`, `southeast`, `southwest`)
- **Data Quality Highlights**:
  - Missing values: 0 (0.0%).
  - Duplicates: 1 row reported at index 581 (handled configurably).

---

## 3. Verified Model Evaluation & Comparison

Models were trained on 1,070 training rows using **5-Fold Cross Validation** with fixed random seed ($42$):

| Architecture | CV RMSE ($) | CV RMSE Std ($) | CV MAE ($) | CV $R^2$ | Selection Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **$4,620.94** | **$583.60** | **$2,571.20** | **0.8508** | **★ WINNER (Champion)** |
| **Gradient Boosting Regressor** | $4,622.85 | $556.35 | $2,599.21 | 0.8509 | Runner-up |
| **Linear Regression** | $6,123.35 | $491.65 | $4,234.98 | 0.7388 | Baseline |

### Held-Out Test Evaluation Results (Unseen $N=268$)
Evaluated strictly against held-out test data (`data/processed/test.csv`):

- **Test RMSE**: **$4,462.02**
- **Test MAE**: **$2,529.67**
- **Test $R^2$**: **0.8718**
- **Overfitting Delta ($\Delta R^2$)**: **0.0309**

---

## 4. Quality Gate Verification

All quality gates configured in `config/quality_gates.yaml` passed:

| Metric | Condition | Configured Threshold | Actual Value | Gate Status |
| :--- | :--- | :--- | :--- | :--- |
| **$R^2$ Score** | $\ge$ | $0.8000$ | **$0.8718$** | **✓ PASS** |
| **RMSE** | $\le$ | $\$5,200.00$ | **$\$4,462.02$** | **✓ PASS** |
| **MAE** | $\le$ | $\$3,000.00$ | **$\$2,529.67$** | **✓ PASS** |
| **Overfitting Delta ($\Delta R^2$)**| $\le$ | $0.1500$ | **$0.0309$** | **✓ PASS** |

**Promotion Decision**: Model successfully promoted to MLflow Model Registry with alias **`@champion`**.

---

## 5. Statistical Data Drift Monitoring

Evaluated against the reference baseline (`data/reference/reference.csv`, $N=1,070$):

| Feature | Feature Type | Statistical Test | Test Statistic | $p$-value | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | Numerical | Two-Sample Kolmogorov-Smirnov | 0.0488 | 0.6670 | `NO_DRIFT` |
| `bmi` | Numerical | Two-Sample Kolmogorov-Smirnov | 0.0562 | 0.4888 | `NO_DRIFT` |
| `children` | Numerical | Two-Sample Kolmogorov-Smirnov | 0.0189 | 1.0000 | `NO_DRIFT` |
| `sex` | Categorical | Chi-Square Contingency | 0.8892 | 0.3457 | `NO_DRIFT` |
| `smoker` | Categorical | Chi-Square Contingency | 0.0042 | 0.9485 | `NO_DRIFT` |
| `region` | Categorical | Chi-Square Contingency | 2.9395 | 0.4010 | `NO_DRIFT` |

- **Drifted Features**: 0 / 6 (0.0%).
- **Alert Status**: `HEALTHY` (Alert triggered: `False`).

---

## 6. Resilience & Failure Injection Testing

10 failure injection scenarios were evaluated with 100% resilience:
1. Missing Raw Dataset: Caught `FileNotFoundError` (Handled).
2. Invalid Data Schema: Caught `ValueError` (Handled).
3. Missing Model Artifact: Caught `FileNotFoundError` (Handled).
4. Missing Preprocessor Artifact: Caught `FileNotFoundError` (Handled).
5. Invalid Inference Input: Caught `ValueError` (Handled).
6. Quality Gate Rejection: Verified `FAIL` decision blocks bad model.
7. Data Drift Alert: Verified 100% shifted data triggers `DRIFT_ALERT`.
8. API Artifact Unavailable: Verified HTTP `503 Service Unavailable`.
9. Model Registry Gate Enforcement: Verified failing gate halts promotion.
10. Invalid Configuration File: Caught `FileNotFoundError` (Handled).

---

## 7. Test Suite Summary

- **Total Unit & Integration Tests**: **71 tests**
- **Test Pass Rate**: **100% (71 passed in 14.28s)**
- **Code Coverage**: **86% line coverage**
- **PEP 8 Compliance**: **Flake8 (0 warnings, 0 errors)**
