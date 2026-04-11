"""
Feature Engineering — XGBoost v2
=================================
Module standalone. Appelle build_features() pour obtenir X_train, X_test, y, folds.
Cache sauvegardé dans xgboost_v2/cache/features.pkl.

Features incluses (~500+) :
  - Géométrie reprojetée EPSG:3857 (area, perimeter, compactness, convexité, élongation...)
  - Centroïdes lat/lon + bins géographiques (0.05° ≈ 5 km)
  - RGB images : stats temporelles, deltas consécutifs, delta relatif last/first
  - Indices spectraux par date : GRI, BRI, VARI, EXG, RGI, brightness, coeff. variation
  - Status de changement : ordinal, deltas consécutifs, stats globales, patterns Demolition
  - Features temporelles des dates : année, mois cyclique, timestamps, deltas, ordre chrono
  - Features d'interaction et signaux ciblés (Mega Projects, Demolition, Road)
  - Target encoding fold-aware (urban_type, geography_type, change_status, cellule géo)
  - One-hot encoding (urban_type, geography_type, change_status)
  - Features binaires multi-valeurs (urban_type, geography_type)
  - Features spatiales de voisinage (cKDTree : dist, n_voisins, aire moyenne)
  - Imputation par médiane (pas de fillna(0))
"""

import re
import gc
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
SEED    = 42

CHANGE_TYPE_MAP = {
    "Demolition": 0, "Road": 1, "Residential": 2,
    "Commercial": 3, "Industrial": 4, "Mega Projects": 5,
}
INV_MAP      = {v: k for k, v in CHANGE_TYPE_MAP.items()}
TARGET_NAMES = [INV_MAP[i] for i in range(6)]

STATUS_ORDER = {
    "Greenland": 0, "Land Cleared": 1, "Excavation": 2,
    "Materials Dumped": 3, "Materials Introduced": 4,
    "Construction Started": 5, "Construction Midway": 6,
    "Construction Done": 7, "Prior Construction": 8, "Operational": 9,
}


# ─── Target encoding (fold-aware, anti-leakage) ──────────────────────────────
def _target_encode_column(train_col, test_col, y, n_classes, folds, smoothing=10):
    """
    Encode une colonne catégorielle par target encoding.
    Sur le train : valeurs calculées uniquement sur les folds hors validation (anti-leakage).
    Sur le test  : valeurs calculées sur tout le train.
    Retourne train_encoded (n_train, n_classes) et test_encoded (n_test, n_classes).
    """
    global_mean = np.array([(y == c).mean() for c in range(n_classes)])

    train_encoded = np.zeros((len(train_col), n_classes))
    for tr_idx, val_idx in folds:
        for c in range(n_classes):
            stats = pd.DataFrame({
                "cat":    train_col.iloc[tr_idx],
                "target": (y[tr_idx] == c).astype(float),
            })
            agg      = stats.groupby("cat")["target"].agg(["mean", "count"])
            smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean[c]) / (agg["count"] + smoothing)
            train_encoded[val_idx, c] = (
                train_col.iloc[val_idx].map(smoothed).fillna(global_mean[c]).values
            )

    test_encoded = np.zeros((len(test_col), n_classes))
    for c in range(n_classes):
        stats    = pd.DataFrame({"cat": train_col, "target": (y == c).astype(float)})
        agg      = stats.groupby("cat")["target"].agg(["mean", "count"])
        smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean[c]) / (agg["count"] + smoothing)
        test_encoded[:, c] = test_col.map(smoothed).fillna(global_mean[c]).values

    return train_encoded, test_encoded


# ─── Feature engineering de base ─────────────────────────────────────────────
def _engineer_features(df):
    """
    Calcule les features sur un GeoDataFrame (train ou test).
    df doit avoir sa CRS alignée (categories normalisées avant appel).
    """
    feat = pd.DataFrame(index=df.index)

    # ── 1. Géométrie ──────────────────────────────────────────────────────────
    geom_proj = df.geometry.to_crs(epsg=3857)

    feat["area"]               = geom_proj.area
    feat["perimeter"]          = geom_proj.length
    feat["compactness"]        = 4 * np.pi * feat["area"] / (feat["perimeter"] ** 2 + 1e-10)
    feat["area_perimeter_ratio"] = feat["area"] / (feat["perimeter"] + 1e-10)

    bounds = geom_proj.bounds
    feat["bbox_width"]        = bounds["maxx"] - bounds["minx"]
    feat["bbox_height"]       = bounds["maxy"] - bounds["miny"]
    feat["bbox_aspect_ratio"] = feat["bbox_width"] / (feat["bbox_height"] + 1e-10)
    feat["bbox_area"]         = feat["bbox_width"] * feat["bbox_height"]
    feat["fill_ratio"]        = feat["area"] / (feat["bbox_area"] + 1e-10)

    centroids = df.geometry.centroid
    feat["centroid_lon"] = centroids.x
    feat["centroid_lat"] = centroids.y

    # Bins géographiques 0.05° ≈ 5 km — "cette zone est majoritairement X"
    feat["centroid_lon_bin"] = (centroids.x / 0.05).round().astype("int32")
    feat["centroid_lat_bin"] = (centroids.y / 0.05).round().astype("int32")

    feat["n_vertices"] = df.geometry.apply(
        lambda g: len(g.exterior.coords) if hasattr(g, "exterior") and g.exterior is not None else 0
    )
    feat["log_area"]      = np.log1p(feat["area"])
    feat["log_perimeter"] = np.log1p(feat["perimeter"])

    # Convexité : area / convex_hull_area (Road = non-convexe, Mega = très convexe)
    def safe_convex_hull_area(g):
        try:
            return g.convex_hull.area
        except Exception:
            return np.nan

    convex_hull_area = geom_proj.apply(safe_convex_hull_area)
    feat["convexity"] = feat["area"] / (
        convex_hull_area.fillna(convex_hull_area.median()) + 1e-10
    )

    # Élongation : max/min dimension (Road = très allongé)
    feat["elongation"] = (
        np.maximum(feat["bbox_width"], feat["bbox_height"]) /
        (np.minimum(feat["bbox_width"], feat["bbox_height"]) + 1e-10)
    )

    feat["area_per_vertex"]        = feat["area"] / (feat["n_vertices"] + 1e-10)
    feat["log_area_x_compactness"] = feat["log_area"] * feat["compactness"]

    # Buckets de taille absolue (Mega Projects = très grande area)
    feat["area_bucket"] = pd.cut(
        feat["area"],
        bins=[0, 1_000, 5_000, 20_000, 100_000, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    # ── 2. Features RGB des images ────────────────────────────────────────────
    rgb_cols = [c for c in df.columns if c.startswith("img_")]
    for c in rgb_cols:
        feat[c] = df[c].values

    # Flag global de données manquantes
    if rgb_cols:
        feat["has_missing_img"] = df[rgb_cols[0]].isnull().astype(int)
    else:
        feat["has_missing_img"] = 0

    for d in range(1, 6):
        r_mean = f"img_red_mean_date{d}"
        g_mean = f"img_green_mean_date{d}"
        b_mean = f"img_blue_mean_date{d}"
        r_std  = f"img_red_std_date{d}"
        g_std  = f"img_green_std_date{d}"
        b_std  = f"img_blue_std_date{d}"

        if all(c in df.columns for c in [r_mean, g_mean, b_mean]):
            r = df[r_mean]
            g = df[g_mean]
            b = df[b_mean]

            # Statistiques synthétiques par date
            feat[f"brightness_date{d}"] = (r + g + b) / 3.0
            feat[f"color_std_date{d}"]  = df[[r_mean, g_mean, b_mean]].std(axis=1)

            # Indices spectraux
            feat[f"gri_date{d}"]  = (g - r) / (g + r + 1e-10)           # Green-Red Index
            feat[f"bri_date{d}"]  = (b - r) / (b + r + 1e-10)           # Blue-Red Index
            feat[f"vari_date{d}"] = (g - r) / (g + r - b + 1e-10)       # Visible Atm. Resistant Index
            feat[f"exg_date{d}"]  = 2 * g - r - b                        # Excess Green (greenland)
            feat[f"rgi_date{d}"]  = r / (g + 1e-10)                      # Red-Green Index (sol nu)

            # Flag de données manquantes par date
            feat[f"missing_date{d}"] = r.isnull().astype(int)
        else:
            feat[f"brightness_date{d}"] = np.nan
            feat[f"color_std_date{d}"]  = np.nan
            feat[f"gri_date{d}"]        = np.nan
            feat[f"bri_date{d}"]        = np.nan
            feat[f"vari_date{d}"]       = np.nan
            feat[f"exg_date{d}"]        = np.nan
            feat[f"rgi_date{d}"]        = np.nan
            feat[f"missing_date{d}"]    = 1

        if all(c in df.columns for c in [r_std, g_std, b_std]):
            feat[f"avg_std_date{d}"] = df[[r_std, g_std, b_std]].mean(axis=1)
            # Coefficient de variation par canal (texture relative)
            for ch, m_col, s_col in [
                ("red",   r_mean, r_std),
                ("green", g_mean, g_std),
                ("blue",  b_mean, b_std),
            ]:
                m = df[m_col]
                s = df[s_col]
                feat[f"cov_{ch}_date{d}"] = s / (m.abs() + 1e-10)
        else:
            feat[f"avg_std_date{d}"] = np.nan
            for ch in ["red", "green", "blue"]:
                feat[f"cov_{ch}_date{d}"] = np.nan

    # Nombre total de dates manquantes
    feat["n_missing_dates"] = sum(feat[f"missing_date{d}"] for d in range(1, 6))

    # Ratio et delta relatif last/first par canal
    for ch in ["red", "green", "blue"]:
        c1 = f"img_{ch}_mean_date1"
        c5 = f"img_{ch}_mean_date5"
        if c1 in df.columns and c5 in df.columns:
            first = df[c1]
            last  = df[c5]
            feat[f"{ch}_ratio_last_first"]      = last / (first.abs() + 1e-10)
            feat[f"{ch}_rel_change_last_first"]  = (last - first) / (first.abs() + 1.0)
        else:
            feat[f"{ch}_ratio_last_first"]     = np.nan
            feat[f"{ch}_rel_change_last_first"] = np.nan

    # Delta relatif de luminosité
    if "brightness_date1" in feat.columns and "brightness_date5" in feat.columns:
        feat["brightness_rel_change_last_first"] = (
            feat["brightness_date5"] - feat["brightness_date1"]
        ) / (feat["brightness_date1"].abs() + 1.0)
    else:
        feat["brightness_rel_change_last_first"] = np.nan

    # Deltas consécutifs par canal (date d → date d+1)
    for ch in ["red", "green", "blue"]:
        for d in range(1, 5):
            c_d   = f"img_{ch}_mean_date{d}"
            c_d1  = f"img_{ch}_mean_date{d + 1}"
            cs_d  = f"img_{ch}_std_date{d}"
            cs_d1 = f"img_{ch}_std_date{d + 1}"
            if c_d in df.columns and c_d1 in df.columns:
                feat[f"delta_{ch}_mean_d{d}_d{d+1}"] = df[c_d1] - df[c_d]
            else:
                feat[f"delta_{ch}_mean_d{d}_d{d+1}"] = np.nan
            if cs_d in df.columns and cs_d1 in df.columns:
                feat[f"delta_{ch}_std_d{d}_d{d+1}"] = df[cs_d1] - df[cs_d]
            else:
                feat[f"delta_{ch}_std_d{d}_d{d+1}"] = np.nan

    # Stats temporelles globales par canal
    for ch in ["red", "green", "blue"]:
        mean_cols = [f"img_{ch}_mean_date{d}" for d in range(1, 6) if f"img_{ch}_mean_date{d}" in df.columns]
        if mean_cols:
            arr = df[mean_cols]
            feat[f"{ch}_mean_overall_mean"]  = arr.mean(axis=1)
            feat[f"{ch}_mean_overall_std"]   = arr.std(axis=1)
            feat[f"{ch}_mean_overall_range"] = arr.max(axis=1) - arr.min(axis=1)
        else:
            feat[f"{ch}_mean_overall_mean"]  = np.nan
            feat[f"{ch}_mean_overall_std"]   = np.nan
            feat[f"{ch}_mean_overall_range"] = np.nan

    # Stats temporelles de luminosité
    bright_cols = [f"brightness_date{d}" for d in range(1, 6)]
    feat["brightness_overall_mean"]  = feat[bright_cols].mean(axis=1)
    feat["brightness_overall_std"]   = feat[bright_cols].std(axis=1)
    feat["brightness_overall_range"] = feat[bright_cols].max(axis=1) - feat[bright_cols].min(axis=1)

    # ── 3. Statuts de changement ──────────────────────────────────────────────
    status_cols = []
    for d in range(5):
        col = f"change_status_date{d}"
        if col in df.columns:
            raw = df[col]
            if hasattr(raw.dtype, "categories"):
                raw = raw.astype(str)
            feat[f"status_ord_date{d}"] = raw.map(STATUS_ORDER)
            status_cols.append(f"status_ord_date{d}")
        else:
            feat[f"status_ord_date{d}"] = np.nan

    if not status_cols:
        status_cols = [f"status_ord_date{d}" for d in range(5)]

    for d in range(4):
        feat[f"status_delta_{d}_{d+1}"] = (
            feat[f"status_ord_date{d+1}"] - feat[f"status_ord_date{d}"]
        )

    all_status_cols = [f"status_ord_date{d}" for d in range(5)]
    feat["status_max"]      = feat[all_status_cols].max(axis=1)
    feat["status_min"]      = feat[all_status_cols].min(axis=1)
    feat["status_range"]    = feat["status_max"] - feat["status_min"]
    feat["status_mean"]     = feat[all_status_cols].mean(axis=1)
    feat["status_std"]      = feat[all_status_cols].std(axis=1)
    feat["n_status_changes"] = feat[[f"status_delta_{d}_{d+1}" for d in range(4)]].ne(0).sum(axis=1)
    feat["status_first"]    = feat["status_ord_date0"]
    feat["status_last"]     = feat["status_ord_date4"]

    # ── 4. Features de dates ──────────────────────────────────────────────────
    parsed_dates = {}
    for d in range(5):
        col = f"date{d}"
        if col in df.columns:
            parsed_dates[d] = pd.to_datetime(df[col], format="%d-%m-%Y", errors="coerce")
        else:
            parsed_dates[d] = pd.Series(pd.NaT, index=df.index)

        feat[f"date{d}_year"]      = parsed_dates[d].dt.year
        feat[f"date{d}_month"]     = parsed_dates[d].dt.month
        feat[f"date{d}_dayofyear"] = parsed_dates[d].dt.dayofyear
        # Encodage cyclique du mois
        feat[f"date{d}_month_sin"] = np.sin(2 * np.pi * parsed_dates[d].dt.month / 12)
        feat[f"date{d}_month_cos"] = np.cos(2 * np.pi * parsed_dates[d].dt.month / 12)
        ts = parsed_dates[d].astype(np.int64) // 10**9
        ts = ts.astype(float)
        ts[ts < 0] = np.nan
        feat[f"date{d}_timestamp"] = ts

    # Ordre chronologique des dates et deltas temporels
    date_ts_cols = [f"date{d}_timestamp" for d in range(5)]
    date_ts_vals = feat[date_ts_cols].values.copy()
    date_ts_for_sort = np.where(np.isnan(date_ts_vals), np.inf, date_ts_vals)
    sorted_ts = np.sort(date_ts_for_sort, axis=1)
    sorted_ts[sorted_ts == np.inf] = np.nan

    for i in range(5):
        feat[f"chrono_date{i}_ts"] = sorted_ts[:, i]
    for i in range(4):
        feat[f"chrono_time_delta_{i}_{i+1}"] = sorted_ts[:, i + 1] - sorted_ts[:, i]
    feat["total_time_span"] = sorted_ts[:, 4] - sorted_ts[:, 0]
    feat["mean_time_gap"]   = feat["total_time_span"] / 4

    # Statuts dans l'ordre chronologique
    date_ts_for_argsort = np.where(np.isnan(date_ts_vals), np.inf, date_ts_vals)
    chrono_order   = np.argsort(date_ts_for_argsort, axis=1)
    status_vals    = feat[all_status_cols].values
    chrono_status  = np.take_along_axis(status_vals, chrono_order, axis=1)

    for i in range(5):
        feat[f"chrono_status_{i}"] = chrono_status[:, i]
    for i in range(4):
        feat[f"chrono_status_delta_{i}_{i+1}"] = chrono_status[:, i + 1] - chrono_status[:, i]

    feat["chrono_status_total_change"] = chrono_status[:, 4] - chrono_status[:, 0]
    diffs = np.diff(chrono_status, axis=1)
    feat["chrono_status_max_jump"]     = np.nanmax(diffs, axis=1)
    feat["chrono_status_min_jump"]     = np.nanmin(diffs, axis=1)
    feat["chrono_status_is_monotonic"] = (diffs >= 0).all(axis=1).astype(int)

    # ── 5. Features d'interaction ──────────────────────────────────────────────
    feat["area_x_status_range"]       = feat["area"] * feat["status_range"]
    feat["compactness_x_brightness"]  = feat["compactness"] * feat["brightness_overall_mean"]
    feat["n_vertices_x_area"]         = feat["n_vertices"] * feat["log_area"]
    feat["lat_x_lon"]                 = feat["centroid_lat"] * feat["centroid_lon"]
    feat["elongation_x_log_area"]     = feat["elongation"] * feat["log_area"]
    feat["area_x_status_last"]        = feat["area"] * feat["status_last"]
    feat["convexity_x_status_range"]  = feat["convexity"] * feat["status_range"]

    # ── 6. Interactions ciblées classes minoritaires ───────────────────────────
    # RGI dernière date × log(area) → sol nu + grande zone = Mega Projects
    if "rgi_date5" in feat.columns:
        feat["rgi_last_x_log_area"] = feat["rgi_date5"] * feat["log_area"]
    else:
        feat["rgi_last_x_log_area"] = np.nan

    # EXG dernière date × log(area) → vert + grande zone = Mega Projects
    # (exg_date5 : Mega=+6 vs Commercial=-0.7, Industrial=-16 → signal fort)
    if "exg_date5" in feat.columns:
        feat["exg_last_x_log_area"]     = feat["exg_date5"] * feat["log_area"]
        feat["exg_last_x_centroid_lat"] = feat["exg_date5"] * feat["centroid_lat"]
    else:
        feat["exg_last_x_log_area"]     = np.nan
        feat["exg_last_x_centroid_lat"] = np.nan

    # Latitude × log(area) → Industrial : zone géographique + taille
    # (centroid_lat : Industrial=5 vs Commercial=29 → signal géographique fort)
    feat["lat_x_log_area"] = feat["centroid_lat"] * feat["log_area"]

    # EXG moyen sur toutes les dates × log(area)
    exg_cols = [f"exg_date{d}" for d in range(1, 6) if f"exg_date{d}" in feat.columns]
    if exg_cols:
        feat["exg_mean_x_log_area"] = feat[exg_cols].mean(axis=1) * feat["log_area"]
    else:
        feat["exg_mean_x_log_area"] = np.nan

    # Hétérogénéité de texture × log(area) → Industrial (toits variés, parkings)
    cov_cols = [c for c in feat.columns if c.startswith("cov_")]
    feat["cov_mean"]            = feat[cov_cols].mean(axis=1) if cov_cols else np.nan
    feat["cov_mean_x_log_area"] = feat["cov_mean"] * feat["log_area"]

    # Taux de progression du statut (lent sur longue durée = Mega Projects)
    total_time_days          = feat["total_time_span"] / 86400
    feat["status_change_rate"] = feat["status_range"] / (total_time_days + 1)

    # Grande superficie + statut moyen avancé = Mega Projects en chantier
    feat["area_x_status_mean"] = feat["area"] * feat["status_mean"]

    # ── 7. Features ciblées Mega Projects ─────────────────────────────────────
    # Mega Projects = grande aire + rarement "Operational"
    feat["is_operational_last"]   = (feat["chrono_status_4"] == 9).astype(int)
    feat["is_not_operational"]    = (feat["chrono_status_4"] < 9).astype(int)
    feat["status_max_minus_last"] = feat["status_max"] - feat["status_last"]

    feat["large_and_unfinished"] = (
        (feat["area_bucket"] >= 3).astype(int) * feat["is_not_operational"]
    )

    feat["log_area_sq"]          = feat["log_area"] ** 2
    feat["log_area_x_status_max"] = feat["log_area"] * feat["status_max"]

    feat["long_construction"] = (
        total_time_days > total_time_days.quantile(0.90)
    ).astype(int)
    feat["log_area_x_time"]   = feat["log_area"] * np.log1p(total_time_days)

    # ── 8. Features ciblées Démolition ────────────────────────────────────────
    # Démolition = SEULE classe dont le status RÉGRESSE
    # Pattern typique : Prior Construction (8) → ... → Greenland (0)
    feat["starts_prior_construction"] = (feat["chrono_status_0"] == 8).astype(int)
    feat["ends_at_greenland"]         = (feat["chrono_status_4"] == 0).astype(int)
    feat["demolition_pattern"]        = feat["starts_prior_construction"] * feat["ends_at_greenland"]

    feat["prior_construction_count"] = (feat[all_status_cols] == 8).sum(axis=1)
    feat["status_net_change"]        = feat["chrono_status_4"] - feat["chrono_status_0"]
    feat["is_decreasing_status"]     = (feat["status_net_change"] < -3).astype(int)
    feat["prior_x_status_drop"]      = feat["starts_prior_construction"] * feat["status_max_minus_last"]

    # ── 9. Road pattern ───────────────────────────────────────────────────────
    # Road : status bas tout au long (jamais Operational, jamais Prior Construction)
    feat["road_pattern"] = (
        (feat["status_max"] <= 6) & (feat["chrono_status_0"] <= 1)
    ).astype(int)

    # Initialisé à 0 — sera mis à jour après add_multival_features()
    for fname in ["geo_dense_forest", "geo_barren_land", "geo_desert", "geo_river", "geo_farms"]:
        feat[fname] = 0
    feat["dense_forest_x_log_area"] = 0.0
    feat["arid_geo_x_log_area"]     = 0.0

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feat


# ─── One-hot encoding ─────────────────────────────────────────────────────────
def _add_onehot_features(train_gdf, test_gdf, X_train, X_test):
    """
    One-hot encoding de urban_type, geography_type et change_status_date*.
    Calculé sur train+test combinés pour avoir les mêmes colonnes partout.
    """
    combined = pd.concat([train_gdf, test_gdf], ignore_index=True)
    n_train  = len(train_gdf)

    for col in ["urban_type", "geography_type"]:
        fill = "NSP_urban" if col == "urban_type" else "NSP_geo"
        vals    = combined[col].astype(str).replace("nan", fill)
        dummies = pd.get_dummies(vals, prefix=f"oh_{col}", dtype=int)
        for c in dummies.columns:
            X_train[c] = dummies[c].iloc[:n_train].values
            X_test[c]  = dummies[c].iloc[n_train:].values

    for d in range(5):
        col  = f"change_status_date{d}"
        vals = combined[col].astype(str).replace("nan", "NSP_status")
        dummies = pd.get_dummies(vals, prefix=f"oh_{col}", dtype=int)
        for c in dummies.columns:
            X_train[c] = dummies[c].iloc[:n_train].values
            X_test[c]  = dummies[c].iloc[n_train:].values

    return X_train, X_test


# ─── Multi-value features ─────────────────────────────────────────────────────
def _add_multival_features(train_gdf, test_gdf, X_train, X_test):
    """
    Features binaires pour chaque valeur atomique de urban_type et geography_type.
    Ex : "Sparse Urban,Industrial" → mv_urban_type_Sparse_Urban=1, mv_urban_type_Industrial=1
    "N,A" est normalisé en "NA" pour éviter les faux atomes "N" et "A".
    """
    n_train  = len(train_gdf)
    combined = pd.concat([train_gdf, test_gdf], ignore_index=True)

    for col in ["urban_type", "geography_type"]:
        vals = (
            combined[col]
            .astype(str)
            .replace("nan", "NA")
            .str.replace(r"(?<![A-Za-z])N,A(?![A-Za-z])", "NA", regex=True)
        )
        split_vals = vals.str.split(",").apply(
            lambda x: [v.strip() for v in x if v.strip()]
        )

        all_atoms = sorted({v for row in split_vals for v in row if v != "NA"})

        for atom in all_atoms:
            safe_name = re.sub(r"[^A-Za-z0-9_]", "_", atom)
            feat_name = f"mv_{col}_{safe_name}"
            presence  = split_vals.apply(lambda x: int(atom in x)).values
            X_train[feat_name] = presence[:n_train]
            X_test[feat_name]  = presence[n_train:]

    return X_train, X_test


# ─── Features de voisinage spatial ────────────────────────────────────────────
def _add_spatial_features(train_gdf, test_gdf, X_train, X_test):
    """
    Features de densité/voisinage calculées sur train+test combinés (non biaisé).
    Utilise cKDTree sur centroides projetés EPSG:3857.

    Features créées :
      dist_nearest_neighbor       : distance (m) au centroïde le plus proche
      n_neighbors_500m            : nb de polygones à ≤500 m
      n_neighbors_1km             : nb de polygones à ≤1 km
      mean_area_neighbors_1km     : superficie moyenne des voisins à ≤1 km
      std_area_neighbors_1km      : écart-type des superficies voisins à ≤1 km
      n_large_neighbors_1km       : nb de voisins > 5 000 m² à ≤1 km
      area_vs_mean_neighbor_ratio : superficie propre / moyenne voisins
      area_global_rank_pct        : rang percentile de superficie (1.0 = le plus grand)
    """
    n_train = len(train_gdf)

    combined_geom = pd.concat(
        [train_gdf[["geometry"]], test_gdf[["geometry"]]], ignore_index=True
    )
    gdf_proj  = gpd.GeoDataFrame(combined_geom, geometry="geometry",
                                  crs=train_gdf.crs).to_crs(epsg=3857)
    centroids = gdf_proj.geometry.centroid
    cx = centroids.x.values.astype(float)
    cy = centroids.y.values.astype(float)
    areas = gdf_proj.geometry.area.values.astype(float)

    # Remplace les coordonnées invalides par la médiane
    valid_mask = np.isfinite(cx) & np.isfinite(cy)
    if not valid_mask.all():
        med_x, med_y = np.nanmedian(cx), np.nanmedian(cy)
        cx = np.where(valid_mask, cx, med_x)
        cy = np.where(valid_mask, cy, med_y)

    areas = np.where(
        np.isfinite(areas) & (areas > 0),
        areas,
        np.nanmedian(areas[areas > 0])
    )

    xy   = np.column_stack([cx, cy])
    tree = cKDTree(xy)

    # Distance au voisin le plus proche (k=2 : [self=0, nearest=1])
    dists, _     = tree.query(xy, k=2)
    dist_nearest = dists[:, 1]

    # Comptage dans les rayons 500 m et 1 km
    idx_500  = tree.query_ball_point(xy, r=500)
    idx_1000 = tree.query_ball_point(xy, r=1000)

    n_500  = np.array([len(idx) - 1 for idx in idx_500],  dtype=float)
    n_1000 = np.array([len(idx) - 1 for idx in idx_1000], dtype=float)

    global_median_area = np.median(areas)

    mean_area_1000 = np.empty(len(idx_1000))
    std_area_1000  = np.empty(len(idx_1000))
    n_large_1000   = np.empty(len(idx_1000))

    for i, idx in enumerate(idx_1000):
        neighbors = [j for j in idx if j != i]
        if neighbors:
            na = areas[neighbors]
            mean_area_1000[i] = na.mean()
            std_area_1000[i]  = na.std()
            n_large_1000[i]   = (na > 5_000).sum()
        else:
            mean_area_1000[i] = global_median_area
            std_area_1000[i]  = 0.0
            n_large_1000[i]   = 0.0

    area_vs_mean_neighbor = areas / (mean_area_1000 + 1e-6)
    area_global_rank      = rankdata(areas, method="average") / len(areas)

    # Features dérivées — séparation Industrial vs Commercial
    # Industrial : n_neighbors_1km ~2000 vs Commercial ~550 (Cohen d=0.77)
    area_per_neighbor   = areas / (n_1000 + 1)           # petit si zone dense
    neighbor_density_r  = n_500 / (n_1000 + 1)           # proportion voisins proches
    log_n_neighbors     = np.log1p(n_1000)
    large_neighbor_pct  = n_large_1000 / (n_1000 + 1)    # fraction grands voisins
    std_vs_mean_area    = np.where(                       # hétérogénéité des voisins
        mean_area_1000 > 0,
        std_area_1000 / (mean_area_1000 + 1e-6),
        0.0,
    )

    for X, sl in [(X_train, slice(None, n_train)), (X_test, slice(n_train, None))]:
        X["dist_nearest_neighbor"]       = dist_nearest[sl]
        X["n_neighbors_500m"]            = n_500[sl]
        X["n_neighbors_1km"]             = n_1000[sl]
        X["mean_area_neighbors_1km"]     = mean_area_1000[sl]
        X["std_area_neighbors_1km"]      = std_area_1000[sl]
        X["n_large_neighbors_1km"]       = n_large_1000[sl]
        X["area_vs_mean_neighbor_ratio"] = area_vs_mean_neighbor[sl]
        X["area_global_rank_pct"]        = area_global_rank[sl]
        # Nouvelles features Industrial vs Commercial
        X["area_per_neighbor_1km"]       = area_per_neighbor[sl]
        X["neighbor_density_ratio"]      = neighbor_density_r[sl]
        X["log_n_neighbors_1km"]         = log_n_neighbors[sl]
        X["large_neighbor_pct_1km"]      = large_neighbor_pct[sl]
        X["std_vs_mean_area_neighbors"]  = std_vs_mean_area[sl]

    return X_train, X_test


# ─── Point d'entrée principal ─────────────────────────────────────────────────
def build_features(force_rebuild: bool = False) -> dict:
    """
    Construit (ou charge depuis le cache) les features complètes.

    Retourne un dict avec :
      "X_train" : DataFrame des features train (imputation médiane, colonnes sûres)
      "X_test"  : DataFrame des features test
      "y"       : np.ndarray des labels (int, 0-5)
      "folds"   : liste de 5 (train_idx, val_idx)
    """
    cache_path = CACHE_DIR / "features.pkl"

    if cache_path.exists() and not force_rebuild:
        print(f"  [cache] Chargement depuis {cache_path.name}...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print(f"  X_train: {data['X_train'].shape} | X_test: {data['X_test'].shape}")
        return data

    t0 = time.time()
    print("=" * 70)
    print("Feature Engineering — XGBoost v2")
    print("=" * 70)

    # ── Chargement des données brutes ────────────────────────────────────────
    print("\n1) Chargement des données...")
    train_gdf = gpd.read_file(DATA_DIR / "train.geojson")
    test_gdf  = gpd.read_file(DATA_DIR / "test.geojson")
    print(f"  Train: {train_gdf.shape} | Test: {test_gdf.shape}")

    y = train_gdf["change_type"].map(CHANGE_TYPE_MAP).values

    # Aligner les catégories train/test (nécessaire pour one-hot stable)
    combined = pd.concat([train_gdf, test_gdf], ignore_index=True)
    cat_cols  = ["urban_type", "geography_type"] + [f"change_status_date{d}" for d in range(5)]
    for col in cat_cols:
        if col in combined.columns:
            combined[col]    = combined[col].astype("category")
            train_gdf[col]   = pd.Categorical(train_gdf[col], categories=combined[col].cat.categories)
            test_gdf[col]    = pd.Categorical(test_gdf[col],  categories=combined[col].cat.categories)

    # ── Feature engineering de base ──────────────────────────────────────────
    print("\n2) Feature engineering de base...")
    X_train = _engineer_features(train_gdf)
    X_test  = _engineer_features(test_gdf)
    print(f"  Features de base: {X_train.shape[1]}")

    # ── Folds stratifiés ─────────────────────────────────────────────────────
    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    skf_folds = list(skf.split(X_train, y))

    # ── Target encoding (fold-aware) ─────────────────────────────────────────
    print("\n3) Target encoding (fold-aware)...")
    for col_name in ["geography_type", "urban_type"]:
        te_tr, te_te = _target_encode_column(
            train_gdf[col_name].astype(str), test_gdf[col_name].astype(str),
            y, 6, skf_folds, smoothing=10,
        )
        for c in range(6):
            X_train[f"te_{col_name}_class{c}"] = te_tr[:, c]
            X_test[f"te_{col_name}_class{c}"]  = te_te[:, c]

    for d in range(5):
        col_name = f"change_status_date{d}"
        te_tr, te_te = _target_encode_column(
            train_gdf[col_name].astype(str), test_gdf[col_name].astype(str),
            y, 6, skf_folds, smoothing=10,
        )
        for c in range(6):
            X_train[f"te_{col_name}_class{c}"] = te_tr[:, c]
            X_test[f"te_{col_name}_class{c}"]  = te_te[:, c]

    # Target encoding de la cellule géographique lat/lon
    geo_cell_train = (X_train["centroid_lat_bin"].astype(str) + "_"
                      + X_train["centroid_lon_bin"].astype(str))
    geo_cell_test  = (X_test["centroid_lat_bin"].astype(str) + "_"
                      + X_test["centroid_lon_bin"].astype(str))
    te_tr, te_te = _target_encode_column(
        geo_cell_train, geo_cell_test, y, 6, skf_folds, smoothing=5,
    )
    for c in range(6):
        X_train[f"te_geo_cell_class{c}"] = te_tr[:, c]
        X_test[f"te_geo_cell_class{c}"]  = te_te[:, c]

    print(f"  Après target encoding: {X_train.shape[1]}")

    # ── One-hot encoding ─────────────────────────────────────────────────────
    print("\n4) One-hot encoding...")
    X_train, X_test = _add_onehot_features(train_gdf, test_gdf, X_train, X_test)
    print(f"  Après one-hot: {X_train.shape[1]}")

    # ── Features binaires multi-valeurs ──────────────────────────────────────
    print("\n5) Multi-value features...")
    X_train, X_test = _add_multival_features(train_gdf, test_gdf, X_train, X_test)
    print(f"  Après multi-value: {X_train.shape[1]}")

    # ── Interactions géographiques (nécessite les colonnes mv_ créées juste au-dessus) ──
    print("\n6) Geo interactions...")
    for X in [X_train, X_test]:
        for geo_col, feat_name in [
            ("Dense_Forest", "geo_dense_forest"),
            ("Barren_Land",  "geo_barren_land"),
            ("Desert",       "geo_desert"),
            ("River",        "geo_river"),
            ("Farms",        "geo_farms"),
        ]:
            mv_col = f"mv_geography_type_{geo_col}"
            X[feat_name] = X[mv_col].values if mv_col in X.columns else 0

        X["dense_forest_x_log_area"] = X["geo_dense_forest"] * X["log_area"]
        X["arid_geo_x_log_area"]     = (
            X["geo_barren_land"] + X["geo_desert"]
        ).clip(0, 1) * X["log_area"]

        ind_urban = (
            X["mv_urban_type_Industrial"].values
            if "mv_urban_type_Industrial" in X.columns
            else np.zeros(len(X))
        )
        area_rank = (
            X["area_global_rank_pct"].values
            if "area_global_rank_pct" in X.columns
            else np.zeros(len(X))
        )
        X["industrial_urban_x_log_area"]  = ind_urban * X["log_area"].values
        X["industrial_urban_x_area_rank"] = ind_urban * area_rank

    print(f"  Après geo interactions: {X_train.shape[1]}")

    # ── Features de voisinage spatial ─────────────────────────────────────────
    print("\n7) Features spatiales (voisinage cKDTree)...")
    X_train, X_test = _add_spatial_features(train_gdf, test_gdf, X_train, X_test)
    print(f"  Après spatial: {X_train.shape[1]}")

    # Interaction "top 10% des superficies vs voisinage" (calculée après spatial)
    for X in [X_train, X_test]:
        if "area_vs_mean_neighbor_ratio" in X.columns:
            X["is_large_vs_neighbors"] = (
                X["area_vs_mean_neighbor_ratio"].values
                > np.percentile(X["area_vs_mean_neighbor_ratio"].values, 90)
            ).astype(int)
        else:
            X["is_large_vs_neighbors"] = 0

    print(f"  Après spatial+interaction: {X_train.shape[1]}")

    # ── Interactions Industrial vs Commercial (post-spatial) ──────────────────
    print("\n7b) Interactions Industrial vs Commercial...")
    for X in [X_train, X_test]:
        n_nbr  = X["n_neighbors_1km"].values
        log_a  = X["log_area"].values
        lat    = X["centroid_lat"].values
        cmpct  = X["compactness"].values
        s_last = X["status_last"].values

        # Zone dense + grande surface → Industrial
        X["log_area_x_n_neighbors"]     = log_a * n_nbr
        # Zone dense + compacte → Industrial (bâtiments rectangulaires rapprochés)
        X["compactness_x_n_neighbors"]  = cmpct * n_nbr
        # Latitude × densité → discrimine Industrial (lat basse) des Commercial (lat haute)
        X["lat_x_n_neighbors"]          = lat * n_nbr
        # Statut final Operational dans zone dense → Industrial fonctionnel
        X["neighbors_x_status_last"]    = n_nbr * s_last
        # Voisinage rapproché × grande surface → cluster industriel
        X["n500_x_log_area"]            = X["n_neighbors_500m"].values * log_a
        # Distance au voisin × densité → isolé ou en cluster ?
        X["dist_x_log_n_neighbors"]     = (
            np.log1p(X["dist_nearest_neighbor"].values) * X["log_n_neighbors_1km"].values
        )

    print(f"  Après interactions Industrial: {X_train.shape[1]}")

    # ── Nettoyage final ───────────────────────────────────────────────────────
    print("\n8) Nettoyage et imputation...")

    # Supprimer les colonnes sources redondantes déjà encodées
    drop_cols = (
        ["urban_type", "geography_type"]
        + [f"change_status_cat_date{d}" for d in range(5)]
    )
    X_train.drop(columns=[c for c in drop_cols if c in X_train.columns], inplace=True)
    X_test.drop( columns=[c for c in drop_cols if c in X_test.columns],  inplace=True)

    # Imputation par médiane (robuste aux outliers, mieux que 0)
    medians        = X_train.median()
    X_train_filled = X_train.fillna(medians)
    X_test_filled  = X_test.fillna(medians)

    # Noms de colonnes sûrs (XGBoost n'aime pas les caractères spéciaux)
    safe_cols = [re.sub(r"[^A-Za-z0-9_]", "_", f) for f in X_train_filled.columns]
    X_train_filled.columns = safe_cols
    X_test_filled.columns  = safe_cols

    print(f"  Features finales: {X_train_filled.shape[1]}")
    print(f"  Feature engineering terminé en {time.time() - t0:.1f}s")

    # ── Sauvegarde cache ──────────────────────────────────────────────────────
    data = {
        "X_train": X_train_filled,
        "X_test":  X_test_filled,
        "y":       y,
        "folds":   skf_folds,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  [cache] Sauvegardé: {cache_path}")

    del train_gdf, test_gdf, combined, X_train, X_test
    gc.collect()

    return data


if __name__ == "__main__":
    data = build_features(force_rebuild=False)
    print(f"\nX_train : {data['X_train'].shape}")
    print(f"X_test  : {data['X_test'].shape}")
    print(f"y       : {data['y'].shape} | classes: {np.unique(data['y'], return_counts=True)}")
