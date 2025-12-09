from typing import Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import Predictor

app = FastAPI(title="Home Credit Default Risk API", version="0.1.0")

# Lazy load predictor to avoid startup failure if model not available
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = Predictor()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available. Please train the model first. Error: {str(e)}"
            )
    return predictor


class PredictRequest(BaseModel):
    data: Any  # dict or list of dicts
    threshold: float = 0.5


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pred = get_predictor()
    try:
        preds, probs = pred.predict(req.data, threshold=req.threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictResponse(predictions=preds, probabilities=probs)
