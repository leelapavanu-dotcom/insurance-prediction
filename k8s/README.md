# Kubernetes Deployment Guide for Insurance MLOps API

This directory contains production-ready Kubernetes manifests to deploy the **Insurance Charges Prediction FastAPI Service** on local or cloud-native Kubernetes clusters (such as **Minikube**, **Docker Desktop Kubernetes**, **kind**, or cloud providers like **EKS/GKE/AKS**).

---

## 1. Architecture Overview

```
                        [ Ingress / Client ]
                                 │
                                 ▼
                     [ Service: ClusterIP ]
                     (Port 80 -> Target 8000)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [ Pod Replica 1 ]               [ Pod Replica 2 ]
    (FastAPI + Model Artifacts)     (FastAPI + Model Artifacts)
      Config: insurance-api-config    Config: insurance-api-config
```

### Manifests:
- **`configmap.yaml`**: Non-sensitive operational configurations (logging, port, model metadata).
- **`deployment.yaml`**: Manages 2 replicas with rolling updates, resource limits, non-root security context (`UID 10001`), and liveness/readiness probes (`/health`).
- **`service.yaml`**: Internal `ClusterIP` load balancer routing traffic to healthy pods.

---

## 2. Prerequisites

1. **Docker Desktop** (with Kubernetes enabled) or **Minikube**:
   ```bash
   minikube start --driver=docker
   ```
2. **Local Image Loading** (if using Minikube):
   ```bash
   # Build the container image first
   docker build -t insurance-mlops-api:latest .

   # Load image directly into Minikube VM
   minikube image load insurance-mlops-api:latest
   ```

---

## 3. Step-by-Step Deployment

### Step A: Apply ConfigMap
```bash
kubectl apply -f k8s/configmap.yaml
```

### Step B: Apply Deployment
```bash
kubectl apply -f k8s/deployment.yaml
```

### Step C: Apply Service
```bash
kubectl apply -f k8s/service.yaml
```

### Step D: Check Rollout and Pod Status
```bash
kubectl get pods -l app=insurance-api
kubectl rollout status deployment/insurance-api-deployment
```

---

## 4. Testing the Service Locally

Use port-forwarding to access the ClusterIP service from your host machine:

```bash
kubectl port-forward service/insurance-api-service 8080:80
```

In a new terminal window:

### 1. Health Probe
```bash
curl -X GET http://localhost:8080/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "model_name": "random_forest",
  "model_version": "champion",
  "artifacts_loaded": true
}
```

### 2. Predict Medical Charges
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": "female",
    "bmi": 28.5,
    "children": 2,
    "smoker": "no",
    "region": "southwest"
  }'
```

---

## 5. Scaling and Maintenance

### Horizontal Scaling
Scale the deployment dynamically:
```bash
kubectl scale deployment/insurance-api-deployment --replicas=4
```

### View Application Logs
```bash
kubectl logs -l app=insurance-api --tail=100 -f
```

### Cleanup
```bash
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
kubectl delete -f k8s/configmap.yaml
```
