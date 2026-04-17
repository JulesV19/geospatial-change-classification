# Land-Use Change Classification from Satellite Imagery

Multi-class classification of geographic polygons from multi-date satellite imagery.  
Course competition — CentraleSupélec · Jan–Feb 2026 · **Private leaderboard: 0.918 macro F1** (baseline kNN: 0.40)

---

## Problem

Given a geographic polygon observed across **5 satellite dates**, classify its land-use change type into one of 6 categories. The dataset is a derived version of [QFabric (CVPR 2021)](https://sagarverma.github.io/qfabric) — polygons are described by pre-aggregated RGB statistics per date, not raw pixel tiles.

| Class | Train samples | Share |
|---|---|---|
| Residential | 148,435 | 50.1% |
| Commercial | 100,422 | 33.9% |
| Demolition | 31,509 | 10.6% |
| Road | 14,305 | 4.8% |
| Industrial | 1,324 | 0.4% |
| Mega Projects | 151 | 0.1% |

**Evaluation:** macro F1-score — rare classes penalise equally to majority ones.

---

## Approach

### 1. Feature Engineering (~614 features)

Hand-crafted from polygon geometry, 5-date RGB statistics, construction status labels, and spatial context.

**Geometric** — area, perimeter, compactness (4πA/P²), convexity, elongation, log-area, area bucket, rank percentile.

**Spectral** (per date × 5 dates) — RGB means/stds, spectral indices (EXG, RGI, GRI, BRI, VARI), temporal deltas, overall min/max/std across dates.

**Temporal** — dates sorted chronologically (raw data is unordered). Status progression features: net change, monotonicity, max jumps, reversals. Targeted patterns for Demolition and Road.

**Spatial neighbourhood** (cKDTree on EPSG:3857) — neighbour count within 500m and 1km, area statistics, zone-level EXG and brightness of neighbours. Industrial zones have ~4× more neighbours per km² than Commercial (Cohen d = 0.77).

**Encoding** — one-hot, multi-value binary, and fold-aware target encoding for urban type, geography type, and construction status. Target encoding computed on out-of-fold data only.

**Interactions** — cross-products from TP/FN analysis: `exg × log_area`, `n_neighbors × log_area`, `zone_exg_signal`, `bright × neg_exg`.

**Design choice — no geographic coordinates.** Polygon latitude/longitude are available but deliberately excluded. Land-use type is spatially autocorrelated (industrial districts cluster geographically), so including coordinates would give a strong but spurious signal: the model would memorise which locations correspond to which class rather than learning transferable spectral and structural patterns. Including coordinates can push the private score above 0.9; excluding them is the honest measure of what the model has actually learned.

### 2. Model

**XGBoost** `multi:softprob`, histogram method.

```python
xgb.XGBClassifier(
    max_depth=7, min_child_weight=6, gamma=0.5, reg_lambda=2.0,
    subsample=0.9, colsample_bytree=0.9,
    learning_rate=0.05, n_estimators=3000,
    early_stopping_rounds=50, eval_metric="mlogloss",
)
```

**Class imbalance** — manual sample weights (`Industrial ×10`, `Mega ×35`) combined with oversampling within each fold's train split (`MEGA_REPEAT=12`, `INDUSTRIAL_REPEAT=8`), built strictly on training indices.

**Full retrain** — `median(best_iters)` across folds, same oversampling as CV.

### 3. Validation

5-fold stratified CV, OOF predictions. OOF macro F1 is the primary signal — the Kaggle public leaderboard was not used to tune hyperparameters.

---

## Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Demolition | 0.767 | 0.933 | 0.842 | 31,509 |
| Road | 0.830 | 0.795 | 0.812 | 14,305 |
| Residential | 0.823 | 0.812 | 0.818 | 148,435 |
| Commercial | 0.738 | 0.712 | 0.725 | 100,422 |
| Industrial | 0.291 | 0.161 | 0.207 | 1,324 |
| Mega Projects | 0.120 | 0.020 | 0.034 | 151 |
| **Macro F1 (OOF)** | | | **0.573 ± 0.011** | |

**Kaggle:** public 0.845 · private **0.918**. The OOF → private gap reflects that the test set yields different results from the training distribution.

---

## Limitations

**Pre-aggregated input.** The dataset provides mean/std RGB per polygon per date — not pixel tiles. Industrial and Mega Projects are visually distinctive (metallic roofs, construction scale) but those textures vanish once aggregated. This sets a hard ceiling for any tabular approach on these two classes.

**Industrial → Commercial (main bottleneck).** 58.6% of Industrial false negatives are predicted as Commercial with mean confidence 0.766. Both classes follow identical construction status progressions and overlap spectrally at the aggregated level. The QFabric paper notes annotator confusion between them. Neighbourhood density partially discriminates (Industrial zones are ~4× denser) but is insufficient against a 75× class imbalance.

**Mega Projects.** Median OOF probability on true Mega examples: 0.0009. With 151 training examples and no pixel texture, the class is effectively invisible in CV conditions.

**Commercial ↔ Residential.** 24% mutual confusion by volume — both classes share timelines and geographic contexts.

---

## Reproducing

```bash
pip install -r requirements.txt
# Place train.geojson and test.geojson in data/
python feature_engineering.py   # ~4 min — builds feature cache
python analyze_model.py         # ~50 min — OOF CV + full retrain
```

---

## File Structure

```
├── feature_engineering.py    # Feature construction pipeline
├── analyze_model.py          # 5-fold OOF CV + error analysis + full retrain
├── requirements.txt
├── data/                     # Raw GeoJSON files (not tracked)
└── cache/                    # Feature cache (not tracked)
```
