"""SmartCart customer-segmentation API.

POST /predict  -> assigns a customer to one of four KMeans segments: which one,
how confident, and which of their inputs placed them there.
GET  /health   -> liveness + how many segments are loaded.
GET  /segments -> name / description / full profile for every segment.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# The pickled preprocessor references pipeline_utils.FeatureEngineer.
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_utils import RAW_INPUT_COLS, cluster_confidence  # noqa: E402

DEFAULT_ORIGINS = ",".join([
    "https://dweep1128.github.io",  # GitHub Pages (project pages share this origin)
    "http://localhost:8000", "http://127.0.0.1:8000",
    "http://localhost:5500", "http://127.0.0.1:5500",
])
N_DRIVERS = 3  # feature comparisons shown per prediction

model = joblib.load(ARTIFACTS_DIR / "model.joblib")
preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
segments: dict[str, dict] = json.loads((ARTIFACTS_DIR / "segments.json").read_text())

_columns = json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text())
NUMERIC_FEATURES: list[str] = _columns["numeric_features"]
CENTROIDS_SCALED = np.asarray(_columns["centroids_scaled"])  # (n_clusters, 18)
SCALED_PIPELINE = preprocessor[:-1]  # everything except the final PCA step
ENGINEER = preprocessor.named_steps["engineer"]

# Human labels for the numeric model features, and how to render a value.
DRIVER_LABELS = {
    "Income": "Income",
    "Recency": "Days since last purchase",
    "NumDealsPurchases": "Deal purchases",
    "NumWebPurchases": "Web purchases",
    "NumCatalogPurchases": "Catalog purchases",
    "NumStorePurchases": "In-store purchases",
    "NumWebVisitsMonth": "Site visits per month",
    "Complain": "Complained recently",
    "Response": "Accepted the last campaign",
    "Age": "Age",
    "Customer_Tenure_Days": "Days as a member",
    "Total_Spending": "Total spending",
    "Total_Children": "Children at home",
}
_CURRENCY = {"Income", "Total_Spending"}
_BINARY = {"Complain", "Response"}
_WHOLE = {"Age", "Recency", "Customer_Tenure_Days"}


def _fmt(feature: str, value: float, *, is_customer: bool) -> str:
    if feature in _CURRENCY:
        return f"${value:,.0f}"
    if feature in _BINARY:
        return ("yes" if value >= 0.5 else "no") if is_customer else f"{value * 100:.0f}% of segment"
    if feature in _WHOLE:
        return f"{value:.0f}"
    return f"{value:g}" if is_customer else f"{value:.1f}"


def explain_prediction(scaled_row: np.ndarray, engineered_row: pd.Series, label: int) -> list["Driver"]:
    """Top features that placed this customer in `label` rather than the others.

    Score per feature = (mean squared scaled distance to the other centroids)
    - (squared scaled distance to the assigned centroid). Positive means the
    feature pulls toward the assigned segment. Ranked across the numeric model
    features only - the one-hot education / living columns feed the same
    distance but don't read well as "your value vs average".
    """
    others = [k for k in range(CENTROIDS_SCALED.shape[0]) if k != label]
    means = segments[str(label)]["profile"]["feature_means"]
    scored = []
    for i, feature in enumerate(NUMERIC_FEATURES):
        to_assigned = (scaled_row[i] - CENTROIDS_SCALED[label, i]) ** 2
        to_others = np.mean([(scaled_row[i] - CENTROIDS_SCALED[k, i]) ** 2 for k in others])
        scored.append((to_others - to_assigned, feature))

    drivers = []
    for score, feature in sorted(scored, reverse=True)[:N_DRIVERS]:
        if score <= 0:
            break
        drivers.append(Driver(
            feature=DRIVER_LABELS[feature],
            customer=_fmt(feature, float(engineered_row[feature]), is_customer=True),
            segment_average=_fmt(feature, float(means[feature]), is_customer=False),
        ))
    return drivers

app = FastAPI(title="SmartCart Segmentation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Customer(BaseModel):
    """One raw customer record - the CSV schema minus ID."""

    Year_Birth: int = Field(ge=1900, le=2025, examples=[1978])
    Education: Literal["Basic", "2n Cycle", "Graduation", "Master", "PhD"] = Field(examples=["Graduation"])
    Marital_Status: Literal[
        "Single", "Married", "Together", "Divorced", "Widow", "Alone", "Absurd", "YOLO"
    ] = Field(examples=["Married"])
    Income: float = Field(gt=0, le=600_000, examples=[52000])
    Kidhome: int = Field(ge=0, le=10, examples=[0])
    Teenhome: int = Field(ge=0, le=10, examples=[1])
    Dt_Customer: str = Field(examples=["2013-07-15"], description="Signup date, YYYY-MM-DD or DD-MM-YYYY")
    Recency: int = Field(ge=0, le=365, examples=[40])
    MntWines: int = Field(ge=0, examples=[380])
    MntFruits: int = Field(ge=0, examples=[30])
    MntMeatProducts: int = Field(ge=0, examples=[220])
    MntFishProducts: int = Field(ge=0, examples=[45])
    MntSweetProducts: int = Field(ge=0, examples=[30])
    MntGoldProds: int = Field(ge=0, examples=[40])
    NumDealsPurchases: int = Field(ge=0, examples=[2])
    NumWebPurchases: int = Field(ge=0, examples=[5])
    NumCatalogPurchases: int = Field(ge=0, examples=[3])
    NumStorePurchases: int = Field(ge=0, examples=[7])
    NumWebVisitsMonth: int = Field(ge=0, examples=[6])
    Complain: Literal[0, 1] = Field(examples=[0])
    Response: Literal[0, 1] = Field(examples=[0])

    @field_validator("Dt_Customer")
    @classmethod
    def _parseable_date(cls, v: str) -> str:
        if pd.isna(pd.to_datetime(v, dayfirst=True, errors="coerce")):
            raise ValueError("Dt_Customer is not a recognisable date")
        return v


class SegmentScore(BaseModel):
    segment: int
    name: str
    confidence: float


class Driver(BaseModel):
    feature: str
    customer: str
    segment_average: str


class PredictionResponse(BaseModel):
    segment: int
    segment_name: str
    description: str
    confidence: float
    confidence_by_segment: list[SegmentScore]
    drivers: list[Driver]
    segment_profile: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "segments": len(segments)}


@app.get("/segments")
def list_segments() -> dict[str, dict]:
    """Name, description and average profile for every segment - used by the
    frontend to lay out the segment map."""
    return segments


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer) -> PredictionResponse:
    row = pd.DataFrame([customer.model_dump()], columns=RAW_INPUT_COLS)
    try:
        scaled_row = SCALED_PIPELINE.transform(row)[0]
        point = preprocessor.named_steps["pca"].transform(scaled_row.reshape(1, -1))[0]
        engineered_row = ENGINEER.transform(row).iloc[0]
    except Exception as exc:  # noqa: BLE001 - surface preprocessing failure to caller
        raise HTTPException(status_code=422, detail=f"Could not process input: {exc}")

    label = int(model.predict(point.reshape(1, -1))[0])
    confidence = cluster_confidence(point, model.cluster_centers_)

    scores = sorted(
        (
            SegmentScore(
                segment=int(c),
                name=segments[str(c)]["name"],
                confidence=round(float(confidence[c]), 4),
            )
            for c in range(len(confidence))
        ),
        key=lambda s: s.confidence,
        reverse=True,
    )
    return PredictionResponse(
        segment=label,
        segment_name=segments[str(label)]["name"],
        description=segments[str(label)]["description"],
        confidence=round(float(confidence[label]), 4),
        confidence_by_segment=scores,
        drivers=explain_prediction(scaled_row, engineered_row, label),
        segment_profile=segments[str(label)]["profile"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
