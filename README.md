# SmartCart — Customer Segment Finder

Serves the customer-segmentation model from `SmartCart.ipynb` as a FastAPI backend
with a static, build-free frontend. Given one customer's raw record it returns the
KMeans segment they fall into and how confident that call is across all four segments.

## What the model does

The notebook is **unsupervised** — there is no target variable. It engineers features,
one-hot encodes, scales, runs PCA to 3 components, and clusters into `k=4` (chosen by
elbow + silhouette). It fits both KMeans and Agglomerative clustering.

**The served model is KMeans, not the notebook's "winner" (Agglomerative).**
`AgglomerativeClustering` has no `predict` method — it only labels the rows it was fit
on and cannot assign a new customer to a cluster. KMeans is in the notebook, is
deterministic (`random_state=42`), and exposes both `predict` and centroid distances,
which we turn into the confidence score. Segment *shapes* differ slightly between the
two algorithms; the KMeans clusters are profiled below.

### Segment names are human-assigned labels

**The model outputs only a cluster ID (0–3)** plus the distances to each centroid.
The names (`Premium shoppers`, etc.) are our shorthand for each cluster's statistics —
not something the model produces. A guard in `train.py::name_segments` asserts any name
that claims a household structure (`families`, `single-parent`) against the cluster's
actual living-arrangement and children means, so a retrain that reshuffles the clusters
fails loudly instead of shipping a wrong label. The two-sentence description
(spend tier + household shape) is derived from the same stats.

| id | label | avg income | 2-yr spend | children | live w/o partner | in-store | web | catalog | deals | visits/mo | campaign response | n |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Budget-conscious families | $35,605 | $131 | 1.2 | 0% | 3.5 | 2.4 | 0.6 | 2.2 | 6.5 | 6% | 710 |
| 3 | Budget-conscious single-parent households | $38,097 | $183 | 1.2 | 100% | 3.8 | 2.8 | 0.9 | 2.5 | 6.5 | 14% | 458 |
| 1 | Mid-market households | $59,857 | $796 | 1.1 | 18% | 7.6 | 6.3 | 3.5 | 3.6 | 5.6 | 17% | 426 |
| 2 | Premium shoppers | $74,671 | $1,306 | 0.3 | 40% | 8.6 | 5.4 | 5.7 | 1.5 | 3.0 | 24% | 642 |

(`Budget-conscious singles` was renamed to `…single-parent households` — the cluster
averages 1.2 children, so "singles" misdescribed it.) The full per-cluster means live in
`artifacts/segments.json` under `profile.feature_means` and are shown in the UI behind a
"Segment details" toggle.

### Why a customer landed in a segment

`/predict` returns `drivers` — the 3 features that most placed *this* customer in *this*
segment, each with the customer's value next to the segment mean
(`"Catalog purchases": "10" vs "5.7 avg"`). Features are ranked by
`(mean squared scaled distance to the other centroids) − (squared scaled distance to the
assigned centroid)` per feature — positive means the feature pulled toward the assigned
segment. This runs in the 18-d scaled space (what feeds PCA); the ranking is over the 13
numeric features only, since the one-hot education/living columns don't read well as
"your value vs average".

## Pipeline

`pipeline_utils.py` holds the shared `FeatureEngineer` (imported by both `train.py` and
the backend so the pickled preprocessor unpickles at serve time).

1. Impute `Income` with the training median (~$51,382)
2. Engineer: `Age = 2026 - Year_Birth`, `Customer_Tenure_Days` (reference date = max
   signup in training data, **2014-06-29**; later signups clip to 0),
   `Total_Spending` (sum of the six `Mnt*` columns), `Total_Children`
   (`Kidhome + Teenhome`), `Education` 5→3, `Living_With` from `Marital_Status`
3. Drop `ID, Year_Birth, Marital_Status, Kidhome, Teenhome, Dt_Customer`, the six `Mnt*`
4. Training only: drop rows with `Age ≥ 90` or `Income ≥ 600,000` (→ 2,236 rows)
5. `OneHotEncoder(handle_unknown="ignore")` on `Education`, `Living_With`
6. `StandardScaler` on all 18 columns
7. `PCA(n_components=3)`
8. `KMeans(n_clusters=4, random_state=42)`

Confidence = inverse distance to each centroid, normalised to sum to 1.

## Layout

```
train.py               reproduce pipeline, write artifacts/
pipeline_utils.py      shared FeatureEngineer + constants + confidence fn
artifacts/             model.joblib, preprocessor.joblib, feature_columns.json, segments.json
backend/main.py        FastAPI app: POST /predict, GET /health, GET /segments
frontend/              index.html, styles.css, app.js  (no build step)
requirements.txt       pinned runtime deps
runtime.txt / .python-version   3.12.7
```

## Run it

### 1. Artifacts (already committed; regenerate any time)

```bash
python -m pip install -r requirements.txt
python train.py
```

### 2. Backend

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# or: python backend/main.py   (reads PORT, defaults to 8000)
```

- `GET /health` → `{"status":"ok","segments":4}`
- `GET /segments` → name + description + full profile for each cluster
- `POST /predict` → body is the 21-field customer record (see `Customer` in
  `backend/main.py` or `/docs`); returns `segment`, `segment_name`, `description`,
  `confidence`, `confidence_by_segment` (all four), `drivers` (top-3 feature
  comparisons), and `segment_profile` (the assigned cluster's full stats)

Environment (see `.env.example`):

| var | default | meaning |
|---|---|---|
| `PORT` | `8000` | bind port, always on `0.0.0.0` |
| `ALLOWED_ORIGINS` | localhost:8000/5500 (+127.0.0.1) | comma-separated CORS origins |

### 3. Frontend

Serve the folder statically and open it:

```bash
python -m http.server 5500 --directory frontend
# open http://127.0.0.1:5500
```

The API base URL is at the **top of `frontend/app.js`**:

```js
const API_BASE = "http://localhost:8000";   // <-- change to the deployed URL
```

When you point it at a deployed backend, add that frontend's origin to
`ALLOWED_ORIGINS` on the backend.

## Quick check

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "Year_Birth":1975,"Education":"Graduation","Marital_Status":"Married","Income":88000,
  "Kidhome":0,"Teenhome":0,"Dt_Customer":"2013-06-15","Recency":40,"MntWines":900,
  "MntFruits":26,"MntMeatProducts":520,"MntFishProducts":38,"MntSweetProducts":27,
  "MntGoldProds":44,"NumDealsPurchases":2,"NumWebPurchases":4,"NumCatalogPurchases":10,
  "NumStorePurchases":9,"NumWebVisitsMonth":6,"Complain":0,"Response":0}'
# -> "segment": 2, "segment_name": "Premium shoppers", "confidence": 0.51...
```

## Deploy notes

- `requirements.txt` pins `scikit-learn==1.6.1` — needed at load time to unpickle the
  preprocessor, not just to train.
- `runtime.txt` / `.python-version` pin `3.12.7`.
- The backend must run from the project root (or with it importable) so
  `pipeline_utils` resolves during `joblib.load`.
- Free hosts cold-start; the frontend already handles the wake-up delay with one
  automatic retry.
