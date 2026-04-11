"""
Exploratory Data Analysis — Land Use Classification
=====================================================
Génère les figures pour le README GitHub.

Sorties (dossier results/figures/) :
  01_class_distribution.png
  02_geographic_distribution.png
  03_area_distribution.png
  04_status_progression.png
  05_spectral_signatures.png
  06_feature_importance_top20.png
  07_confusion_matrix.png        (si oof_preds.npy existe)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import gaussian_kde

HERE    = Path(__file__).parent
DATA    = HERE / "data"
OUT_DIR = HERE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Demolition", "Road", "Residential", "Commercial", "Industrial", "Mega Projects"]
COLORS  = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c"]
STATUS_ORDER = {
    "Greenland": 0, "Land Cleared": 1, "Excavation": 2,
    "Materials Dumped": 3, "Materials Introduced": 4,
    "Construction Started": 5, "Construction Midway": 6,
    "Construction Done": 7, "Prior Construction": 8, "Operational": 9,
}

print("Chargement des données...")
gdf = gpd.read_file(DATA / "train.geojson")
gdf["class_id"] = gdf["change_type"].map(
    {c: i for i, c in enumerate(CLASSES)}
)
geom_proj   = gdf.geometry.to_crs(epsg=3857)
gdf["area"] = geom_proj.area
gdf["centroid_lat"] = gdf.geometry.centroid.y
gdf["centroid_lon"] = gdf.geometry.centroid.x

for d in range(5):
    col = f"change_status_date{d}"
    gdf[f"status_ord_{d}"] = gdf[col].map(STATUS_ORDER)

print(f"  {len(gdf):,} exemples  |  {gdf['change_type'].nunique()} classes\n")

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. Distribution des classes
# ══════════════════════════════════════════════════════════════════════════════
print("1/7 — Distribution des classes")
counts = gdf["change_type"].value_counts().reindex(CLASSES)

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(CLASSES[::-1], counts[CLASSES[::-1]].values,
               color=COLORS[::-1], edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, counts[CLASSES[::-1]].values):
    pct = 100 * val / len(gdf)
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9)

ax.set_xlabel("Number of samples", fontsize=10)
ax.set_title("Class Distribution — Severe Imbalance\n"
             "Mega Projects (0.1%) and Industrial (0.4%) are rare", fontsize=11)
ax.set_xlim(0, counts.max() * 1.25)
ax.axvline(counts.mean(), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(counts.mean() + 200, -0.6, "mean", color="gray", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "01_class_distribution.png", bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 2. Distribution géographique
# ══════════════════════════════════════════════════════════════════════════════
print("2/7 — Distribution géographique")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Toutes les classes
ax = axes[0]
for i, (cls, col) in enumerate(zip(CLASSES, COLORS)):
    mask = gdf["class_id"] == i
    sub  = gdf[mask].sample(min(3000, mask.sum()), random_state=42)
    ax.scatter(sub["centroid_lon"], sub["centroid_lat"],
               c=col, s=1.5, alpha=0.4, label=cls, rasterized=True)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Geographic Distribution by Class")
ax.legend(markerscale=5, fontsize=8, loc="upper left")

# Zoom Industrial vs Commercial
ax = axes[1]
for cls_i, label, col, size, alpha in [
    (3, "Commercial", COLORS[3], 1.0, 0.25),
    (4, "Industrial", COLORS[4], 8.0, 0.9),
]:
    mask = gdf["class_id"] == cls_i
    sub  = gdf[mask].sample(min(5000, mask.sum()), random_state=42)
    ax.scatter(sub["centroid_lon"], sub["centroid_lat"],
               c=col, s=size, alpha=alpha, label=label, rasterized=True)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Industrial vs Commercial\nGeographic Separation (key discriminating feature)")
ax.legend(markerscale=4, fontsize=9)

fig.suptitle("Spatial Distribution of Land-Use Changes", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "02_geographic_distribution.png", bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 3. Distribution des superficies
# ══════════════════════════════════════════════════════════════════════════════
print("3/7 — Distribution des superficies")
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=False)
log_area_all = np.log1p(gdf["area"])
x_min, x_max = log_area_all.quantile(0.01), log_area_all.quantile(0.99)

for ax, cls, col in zip(axes.flat, CLASSES, COLORS):
    mask    = gdf["class_id"] == CLASSES.index(cls)
    data    = np.log1p(gdf.loc[mask, "area"])
    data_c  = data.clip(x_min, x_max)
    med     = data.median()
    ax.hist(data_c, bins=50, color=col, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(med, color="black", linewidth=1.2, linestyle="--")
    ax.set_title(f"{cls}\n(n={mask.sum():,}  median={np.expm1(med)/1000:.0f}k m²)", fontsize=9)
    ax.set_xlabel("log(area + 1)")
    ax.set_ylabel("Count")

fig.suptitle("Area Distribution by Class  (log scale)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "03_area_distribution.png", bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 4. Progressions de statut typiques
# ══════════════════════════════════════════════════════════════════════════════
print("4/7 — Progressions de statut")

# Trier les dates chronologiquement
date_cols   = [f"date{d}" for d in range(5)]
status_cols = [f"status_ord_{d}" for d in range(5)]
for col in date_cols:
    gdf[col] = pd.to_datetime(gdf[col], format="%d-%m-%Y", errors="coerce")

def chrono_status(row):
    pairs = sorted(
        [(row[f"date{d}"], row[f"status_ord_{d}"]) for d in range(5)
         if pd.notna(row[f"date{d}"])],
        key=lambda x: x[0]
    )
    return [p[1] for p in pairs] if pairs else [np.nan]

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
N_SAMPLE = 300

for ax, cls, col in zip(axes.flat, CLASSES, COLORS):
    mask   = gdf["class_id"] == CLASSES.index(cls)
    sample = gdf[mask].sample(min(N_SAMPLE, mask.sum()), random_state=42)

    for _, row in sample.iterrows():
        seq = chrono_status(row)
        ax.plot(range(len(seq)), seq, color=col, alpha=0.08, linewidth=0.7)

    # Médiane par position
    all_seqs = [chrono_status(row) for _, row in sample.iterrows()]
    max_len  = max(len(s) for s in all_seqs)
    by_pos   = [
        [s[i] for s in all_seqs if i < len(s) and not np.isnan(s[i])]
        for i in range(max_len)
    ]
    medians = [np.median(p) if p else np.nan for p in by_pos]
    ax.plot(range(len(medians)), medians, color="black", linewidth=2, label="Median")

    ax.set_yticks(range(10))
    ax.set_yticklabels(
        ["Greenland", "Cleared", "Excavation", "Mat.Dumped", "Mat.Intro",
         "Constr.Start", "Constr.Mid", "Constr.Done", "Prior Constr.", "Operational"],
        fontsize=6
    )
    ax.set_xlabel("Chronological date index")
    ax.set_title(f"{cls}", fontsize=10)
    ax.set_ylim(-0.5, 9.5)

fig.suptitle("Construction Status Progression Over Time\n"
             "Each line = one polygon (300 samples per class)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "04_status_progression.png", bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 5. Signatures spectrales moyennes
# ══════════════════════════════════════════════════════════════════════════════
print("5/7 — Signatures spectrales")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

for ax, channel, ch_col in zip(axes, ["red", "green", "blue"], ["#e74c3c", "#27ae60", "#2980b9"]):
    for cls, cls_col in zip(CLASSES, COLORS):
        mask     = gdf["class_id"] == CLASSES.index(cls)
        sub      = gdf[mask]
        means    = [sub[f"img_{channel}_mean_date{d}"].mean() for d in range(1, 6)]
        stds     = [sub[f"img_{channel}_mean_date{d}"].std()  for d in range(1, 6)]
        dates    = range(1, 6)
        ax.plot(dates, means, color=cls_col, marker="o", markersize=4, label=cls, linewidth=1.5)
        ax.fill_between(dates,
                        [m - s * 0.3 for m, s in zip(means, stds)],
                        [m + s * 0.3 for m, s in zip(means, stds)],
                        color=cls_col, alpha=0.1)
    ax.set_title(f"{channel.capitalize()} channel")
    ax.set_xlabel("Date index")
    ax.set_ylabel("Mean pixel value")
    ax.set_xticks(range(1, 6))
    if channel == "red":
        ax.legend(fontsize=7, loc="upper left")

fig.suptitle("Mean Spectral Signatures by Class\n"
             "Shaded band = ±0.3 std", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "05_spectral_signatures.png", bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 6. Feature importance top 20
# ══════════════════════════════════════════════════════════════════════════════
print("6/7 — Feature importance")
fi_path = HERE / "feature_importance.csv"
if fi_path.exists():
    fi = pd.read_csv(fi_path).head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(fi["feature"][::-1], fi["gain_pct"][::-1],
                   color="#3498db", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Gain contribution (%)")
    ax.set_title("Top 20 Features by Gain\n"
                 f"Cover {fi['gain_cumul_pct'].iloc[-1]:.1f}% of total gain", fontsize=11)
    for bar, val in zip(bars, fi["gain_pct"][::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=7.5)
    ax.set_xlim(0, fi["gain_pct"].max() * 1.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_feature_importance_top20.png", bbox_inches="tight")
    plt.close()
    print("   feature_importance.csv trouvé — figure générée")
else:
    print("   feature_importance.csv absent — figure ignorée")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Matrice de confusion OOF
# ══════════════════════════════════════════════════════════════════════════════
print("7/7 — Matrice de confusion OOF")
oof_preds_path = HERE / "oof_preds.npy"
oof_proba_path = HERE / "oof_proba.npy"

if oof_preds_path.exists():
    from sklearn.metrics import confusion_matrix, f1_score
    import pickle

    with open(HERE / "cache" / "features.pkl", "rb") as f:
        feat = pickle.load(f)
    y_all     = feat["y"]
    oof_preds = np.load(oof_preds_path)
    cm_norm   = confusion_matrix(y_all, oof_preds, normalize="true")
    f1_per    = f1_score(y_all, oof_preds, average=None, zero_division=0)
    f1_mac    = f1_per.mean()

    short = ["Demo", "Road", "Resi", "Comm", "Indu", "Mega"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(6)); ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(6):
        for j in range(6):
            val = cm_norm[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_title(f"Confusion Matrix — 5-Fold OOF\n"
                 f"Macro F1 = {f1_mac:.3f}  |  per class: "
                 + "  ".join(f"{s}={f1_per[i]:.2f}" for i, s in enumerate(short)),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print(f"   OOF trouvé — F1 macro = {f1_mac:.4f}")
else:
    print("   oof_preds.npy absent — lance analyze_model.py d'abord")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Heatmap top-50 features × classes
# ══════════════════════════════════════════════════════════════════════════════
print("8/8 — Heatmap top-50 features × classes")

imp_path   = HERE / "feature_importance.csv"
feat_cache = HERE / "cache" / "features.pkl"

if imp_path.exists() and feat_cache.exists():
    import pickle
    from scipy.stats import zscore

    df_imp = pd.read_csv(imp_path)
    top50  = df_imp.head(50)["feature"].tolist()

    with open(feat_cache, "rb") as f:
        feat_ckpt = pickle.load(f)

    X = feat_ckpt["X_train"]
    y = feat_ckpt["y"]

    # Garde uniquement les features présentes dans X
    top50 = [f for f in top50 if f in X.columns]

    # Moyenne par classe pour chaque feature
    class_means = np.array([
        X[top50].values[y == c].mean(axis=0)
        for c in range(6)
    ])  # shape (6, n_features)

    # Z-score sur les 6 classes pour chaque feature (axe 0 = classes)
    with np.errstate(invalid="ignore"):
        z = zscore(class_means, axis=0)
    z = np.nan_to_num(z)  # std=0 → zscore=nan → 0

    # Noms raccourcis pour l'axe y
    short_labels = [f[:40] + "…" if len(f) > 40 else f for f in top50]

    fig, ax = plt.subplots(figsize=(10, 18))
    im = ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)

    ax.set_xticks(range(6))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(top50)))
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Class", fontsize=10)
    ax.set_title(
        "Top-50 features — mean z-score per class\n"
        "Red = above average for this class  |  Blue = below average",
        fontsize=10,
    )

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="z-score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_feature_heatmap_top50.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("   08_feature_heatmap_top50.png sauvegardé")
else:
    missing = []
    if not imp_path.exists():
        missing.append("feature_importance.csv (lance feature_importance.py)")
    if not feat_cache.exists():
        missing.append("cache/features.pkl (lance feature_engineering.py)")
    print(f"   Absent — fichiers manquants : {', '.join(missing)}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\nFigures sauvegardées dans : {OUT_DIR}")
print("  01_class_distribution.png")
print("  02_geographic_distribution.png")
print("  03_area_distribution.png")
print("  04_status_progression.png")
print("  05_spectral_signatures.png")
print("  06_feature_importance_top20.png")
print("  07_confusion_matrix.png")
print("  08_feature_heatmap_top50.png")
