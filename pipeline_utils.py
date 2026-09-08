"""Shared pieces for the SmartCart clustering pipeline.

Imported by both `train.py` and the FastAPI backend so the pickled preprocessor
(which references `FeatureEngineer`) can be unpickled at serving time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Notebook: df["Age"] = 2026 - df["Year_Birth"]
AGE_REFERENCE_YEAR = 2026

SPENDING_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]

# Notebook: 5 education levels collapsed to 3
EDUCATION_MAP = {
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate",
}

# Notebook: marital status collapsed to living arrangement
LIVING_WITH_MAP = {
    "Married": "Partner",
    "Together": "Partner",
    "Single": "Alone",
    "Divorced": "Alone",
    "Widow": "Alone",
    "Absurd": "Alone",
    "YOLO": "Alone",
    "Alone": "Alone",
}

# Feature order going into the encoder (matches notebook's df_encoded before OHE).
NUM_COLS = [
    "Income", "Recency", "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
    "Complain", "Response", "Age", "Customer_Tenure_Days",
    "Total_Spending", "Total_Children",
]
CAT_COLS = ["Education", "Living_With"]

# Raw columns the caller must supply (CSV columns minus ID).
RAW_INPUT_COLS = [
    "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
    "Dt_Customer", "Recency", "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumDealsPurchases",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "Complain", "Response",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Reproduces the notebook's pre-encoding steps on raw customer rows.

    Learns the Income median and the tenure reference date from the training
    frame in `fit`, then in `transform`: imputes Income, derives Age, tenure,
    total spend and total children, and collapses Education / Marital_Status.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        income = pd.to_numeric(X["Income"], errors="coerce")
        self.income_median_ = float(income.median())
        self.reference_date_ = pd.to_datetime(X["Dt_Customer"], dayfirst=True).max()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df["Income"] = (
            pd.to_numeric(df["Income"], errors="coerce").fillna(self.income_median_)
        )
        df["Age"] = AGE_REFERENCE_YEAR - df["Year_Birth"]

        signup = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
        tenure = (self.reference_date_ - signup).dt.days
        # ponytail: clip signups later than the training reference date to 0;
        # the notebook never saw future dates. Widen the reference date and
        # retrain if the served population drifts past mid-2014.
        df["Customer_Tenure_Days"] = tenure.clip(lower=0)

        df["Total_Spending"] = df[SPENDING_COLS].sum(axis=1)
        df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
        df["Education"] = df["Education"].map(lambda v: EDUCATION_MAP.get(v, v))
        df["Living_With"] = df["Marital_Status"].map(lambda v: LIVING_WITH_MAP.get(v, "Alone"))

        return df[NUM_COLS + CAT_COLS]


def cluster_confidence(pca_point: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Inverse-distance weights over KMeans centroids, normalised to sum to 1.

    A point sitting on a centroid scores ~1.0 for that segment; a point midway
    between two centroids splits ~50/50 - so the caller can see how close the
    call was.
    """
    distances = np.linalg.norm(centroids - np.asarray(pca_point), axis=1)
    weights = 1.0 / (distances + 1e-9)
    return weights / weights.sum()
