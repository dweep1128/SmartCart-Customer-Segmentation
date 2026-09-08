"""Reproduce the SmartCart notebook pipeline and persist serving artifacts.

Run: ``python train.py``  ->  writes model / preprocessor / column order to
``artifacts/``. The served model is KMeans (it can assign new points);
see README for why the notebook's Agglomerative pick is not used at serving.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline_utils import (
    AGE_REFERENCE_YEAR,
    CAT_COLS,
    NUM_COLS,
    RAW_INPUT_COLS,
    FeatureEngineer,
    cluster_confidence,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "smartcart_customers.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
N_CLUSTERS = 4          # notebook: elbow + silhouette both point to k=4
PCA_COMPONENTS = 3      # notebook: PCA(n_components=3)
MAX_AGE = 90            # notebook outlier rule: keep Age < 90
MAX_INCOME = 600_000    # notebook outlier rule: keep Income < 600_000


def build_preprocessor() -> Pipeline:
    """FeatureEngineer -> one-hot encode -> standard scale -> PCA(3)."""
    return Pipeline([
        ("engineer", FeatureEngineer()),
        ("encode", ColumnTransformer([
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ])),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)),
    ])


def load_training_frame() -> pd.DataFrame:
    """CSV rows that survive the notebook's outlier filters."""
    raw = pd.read_csv(DATA_PATH)
    imputed_income = raw["Income"].fillna(raw["Income"].median())
    age = AGE_REFERENCE_YEAR - raw["Year_Birth"]
    keep = (age < MAX_AGE) & (imputed_income < MAX_INCOME)
    return raw.loc[keep, RAW_INPUT_COLS].reset_index(drop=True)


# Human-assigned labels for the four clusters, ordered by total spend
# (lowest -> highest). The model only emits cluster IDs; these names are our
# shorthand for the statistical profile and are checked against it below, so a
# retrain that shuffles the clusters fails loudly instead of shipping a wrong
# label. Any name making a household-structure claim ("families",
# "single-parent") is asserted against the cluster's living-arrangement and
# children stats in name_segments().
SPEND_RANK_NAMES = [
    "Budget-conscious families",
    "Budget-conscious single-parent households",
    "Mid-market households",
    "Premium shoppers",
]

# One phrase per spend rank (lowest -> highest), for the short description.
TIER_PHRASES = [
    "Lowest income and spending of the four segments.",
    "Below-average income and spending.",
    "Above-average income and spending.",
    "Highest income and spending of the four segments.",
]


def _label_matches_stats(name: str, share_alone: float, children: float) -> bool:
    """Guard: names that claim a household structure must match the cluster."""
    if "single-parent" in name:
        return share_alone >= 0.5 and children >= 0.5
    if "families" in name:
        return share_alone < 0.5 and children >= 0.5
    return True  # "households" / "shoppers" make no such claim


def _household_phrase(share_alone: float, children: float) -> str:
    if share_alone >= 0.5:
        return ("Usually one adult with children at home."
                if children >= 0.5 else "Usually one-adult households.")
    return ("Usually couples with children at home."
            if children >= 0.8 else "Usually couples with few or no children.")


def name_segments(engineered: pd.DataFrame, labels: np.ndarray) -> dict:
    """Name, a short computed description, and the full stat profile per cluster.

    The description is two computed sentences (spend tier + household shape).
    `profile.feature_means` holds the raw mean of every numeric model feature so
    the backend can explain a single prediction against them. Names come from
    SPEND_RANK_NAMES and are asserted against the same stats.
    """
    frame = engineered.assign(
        _cluster=labels, _alone=engineered["Living_With"].eq("Alone").astype(float)
    )
    total = len(frame)
    feat_means = frame.groupby("_cluster")[NUM_COLS].mean()
    children = feat_means["Total_Children"]
    spend = feat_means["Total_Spending"]
    share_alone = frame.groupby("_cluster")["_alone"].mean()
    spend_rank = {c: r for r, c in enumerate(spend.sort_values().index)}

    segments = {}
    for cluster in sorted(int(c) for c in set(labels)):
        rank = spend_rank[cluster]
        name = SPEND_RANK_NAMES[rank]
        assert _label_matches_stats(name, share_alone[cluster], children[cluster]), (
            f"segment label {name!r} contradicts cluster {cluster} stats "
            f"(share_alone={share_alone[cluster]:.2f}, children={children[cluster]:.2f}); "
            f"the retrained clustering has shifted - review SPEND_RANK_NAMES"
        )
        segments[str(cluster)] = {
            "name": name,
            "description": f"{TIER_PHRASES[rank]} "
                           f"{_household_phrase(share_alone[cluster], children[cluster])}",
            "profile": {
                "size": int((labels == cluster).sum()),
                "share_of_base": round(float((labels == cluster).mean()), 3),
                "share_living_alone": round(float(share_alone[cluster]), 3),
                "feature_means": {
                    col: round(float(feat_means.loc[cluster, col]), 2) for col in NUM_COLS
                },
            },
        }
    return segments


def self_check(preprocessor: Pipeline, model: KMeans,
               X_raw: pd.DataFrame, labels: np.ndarray) -> None:
    """Reload the artifacts from disk and confirm they reproduce training output."""
    pp = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    mdl = joblib.load(ARTIFACTS_DIR / "model.joblib")

    sample = X_raw.head(25)
    reloaded = mdl.predict(pp.transform(sample))
    assert np.array_equal(reloaded, labels[:25]), "reloaded artifacts disagree with training"

    conf = cluster_confidence(pp.transform(X_raw.head(1))[0], mdl.cluster_centers_)
    assert len(conf) == N_CLUSTERS, "confidence vector length mismatch"
    assert abs(conf.sum() - 1.0) < 1e-6, "confidence does not sum to 1"
    print("self-check ok")


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    X_raw = load_training_frame()
    print(f"training rows after outlier filter: {len(X_raw)}")

    preprocessor = build_preprocessor()
    X_pca = preprocessor.fit_transform(X_raw)

    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_pca)

    engineered = preprocessor.named_steps["engineer"].transform(X_raw)
    segments = name_segments(engineered, labels)

    encoded_order = [
        name.split("__", 1)[-1]
        for name in preprocessor.named_steps["encode"].get_feature_names_out()
    ]

    # Cluster centres in the scaled 18-d space (pre-PCA). The backend uses these
    # to attribute a single prediction to the features that placed it there.
    scaled = preprocessor[:-1].transform(X_raw)
    centroids_scaled = (
        pd.DataFrame(scaled).assign(_c=labels).groupby("_c").mean()
        .reindex(range(N_CLUSTERS)).values.tolist()
    )

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    (ARTIFACTS_DIR / "feature_columns.json").write_text(json.dumps({
        "raw_input": RAW_INPUT_COLS,
        "engineered": NUM_COLS + CAT_COLS,
        "numeric_features": NUM_COLS,
        "encoded_feature_order": encoded_order,
        "centroids_scaled": centroids_scaled,
        "pca_components": PCA_COMPONENTS,
        "n_clusters": N_CLUSTERS,
        "random_state": RANDOM_STATE,
    }, indent=2))
    (ARTIFACTS_DIR / "segments.json").write_text(json.dumps(segments, indent=2))

    for cid, seg in segments.items():
        print(f"  segment {cid}: {seg['name']} (n={seg['profile']['size']})")

    self_check(preprocessor, model, X_raw, labels)
    print(f"artifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
