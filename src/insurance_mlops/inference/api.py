"""FastAPI REST Service for Insurance Charges Prediction."""

from contextlib import asynccontextmanager
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from insurance_mlops.inference.predictor import (
    BatchPredictionResult,
    InsuranceInputRecord,
    InsurancePredictor,
    SinglePredictionResult,
)
from insurance_mlops.utils.logger import get_logger

logger = get_logger("inference_api")

# Global predictor instance
_predictor: Optional[InsurancePredictor] = None


def get_predictor() -> InsurancePredictor:
    """Retrieves or initializes the singleton InsurancePredictor instance."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = InsurancePredictor()
        except FileNotFoundError as fnf:
            logger.error(f"Failed to load model artifacts: {fnf}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model artifacts not available. Please ensure training pipeline has completed: {fnf}",
            )
        except Exception as exc:
            logger.error(f"Unexpected error loading predictor: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal predictor error: {exc}",
            )
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to pre-load artifacts on startup safely."""
    logger.info("Initializing Insurance Prediction API service...")
    try:
        get_predictor()
        logger.info("Artifacts successfully verified on startup.")
    except HTTPException:
        logger.warning("Artifacts not yet loaded on startup. Endpoints will attempt on-demand retrieval.")
    yield
    logger.info("Shutting down Insurance Prediction API service.")


app = FastAPI(
    title="Insurance Charges Prediction API",
    description=(
        "Production-grade MLOps REST API for estimating healthcare "
        "charges based on demographics and lifestyle factors."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    artifacts_loaded: bool


@app.get("/", tags=["Info"])
def root():
    """Root info endpoint."""
    return {
        "message": "Insurance Charges Prediction API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint verifying artifact availability and model status."""
    try:
        pred = get_predictor()
        return HealthResponse(
            status="healthy",
            model_name=pred.model_name,
            model_version=str(pred.model_version),
            artifacts_loaded=True,
        )
    except HTTPException as e:
        return HealthResponse(
            status=f"degraded: {e.detail}",
            model_name=None,
            model_version=None,
            artifacts_loaded=False,
        )


@app.post(
    "/predict",
    response_model=Union[SinglePredictionResult, BatchPredictionResult],
    tags=["Inference"],
    summary="Predict Medical Insurance Charges",
    description="Accepts a single patient record or a list of patient records and returns predicted charges in USD.",
)
def predict(
    payload: Union[InsuranceInputRecord, List[InsuranceInputRecord]],
):
    """Predicts insurance charges for single or batch patient profiles."""
    pred = get_predictor()

    try:
        # 1. Single Record Request
        if isinstance(payload, InsuranceInputRecord):
            record = payload.model_dump()
            predicted_charge = pred.predict_single(record)
            return SinglePredictionResult(
                predicted_charges=predicted_charge,
                model_name=pred.model_name,
                model_version=str(pred.model_version),
            )

        # 2. Batch List Request
        records = [r.model_dump() for r in payload]
        predictions = pred.predict_records(records)
        return BatchPredictionResult(
            predictions=predictions,
            count=len(predictions),
            model_name=pred.model_name,
            model_version=str(pred.model_version),
        )

    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {exc}",
        )
