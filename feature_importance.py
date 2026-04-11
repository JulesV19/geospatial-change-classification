"""
Classement des features par importance pour XGBoost v2
=======================================================
- Charge les features depuis le cache (feature_engineering.py)
- Entraîne un modèle rapide (80/20 split, LR=0.1, max 1000 rounds avec early stopping)
- Importance par "total_gain" (somme du gain sur tous les splits)
- Sauvegarde :
    feature_importance.csv        — classement complet
    features_to_keep.txt          — features couvrant 99% du gain cumulé
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ─── Config ──────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
CACHE_DIR = HERE / "cache"
SEED      = 42

CUMULATIVE_THRESHOLDS = [0.80, 0.90, 0.95, 0.99, 0.995]

# Poids manuels identiques au pipeline d'entraînement
MANUAL_CW = {0: 1.3, 1: 2.8, 2: 0.27, 3: 0.40, 4: 30.0, 5: 265.0}

TARGET_NAMES = ["Demolition", "Road", "Residential", "Commercial", "Industrial", "Mega Projects"]

# ─── Chargement des features ──────────────────────────────────────────────────
cache_path = CACHE_DIR / "features.pkl"
if not cache_path.exists():
    raise FileNotFoundError(
        f"{cache_path} introuvable. Lance feature_engineering.py d'abord."
    )

print("Chargement des features...")
with open(cache_path, "rb") as f:
    feat_ckpt = pickle.load(f)

X = feat_ckpt["X_train"]
y = feat_ckpt["y"]
print(f"  {X.shape[1]} features | {len(y)} exemples")

# ─── Split 80/20 stratifié ────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
sw_tr = np.array([MANUAL_CW[c] for c in y_tr], dtype=float)

print(f"  Train : {len(X_tr)} | Val : {len(X_val)}")

# ─── Entraînement rapide ──────────────────────────────────────────────────────
print("\nEntraînement rapide XGBoost (LR=0.4, max 1000 rounds, early stopping=50)...")

model = xgb.XGBClassifier(
    objective             = "multi:softprob",
    num_class             = 6,
    eval_metric           = "mlogloss",
    tree_method           = "hist",
    device                = "cpu",
    learning_rate         = 0.4,
    n_estimators          = 1000,
    early_stopping_rounds = 50,
    max_depth             = 10,
    subsample             = 0.9,
    colsample_bytree      = 0.9,
    seed                  = SEED,
    verbosity             = 0,
)

model.fit(
    X_tr, y_tr,
    sample_weight = sw_tr,
    eval_set      = [(X_val, y_val)],
    verbose       = False,
)

print(f"  Arrêt à l'itération {model.best_iteration}")
val_f1 = f1_score(y_val, model.predict(X_val), average="macro")
print(f"  F1 macro val : {val_f1:.5f}")

# ─── Importances par total_gain ───────────────────────────────────────────────
booster = model.get_booster()
scores  = booster.get_score(importance_type="total_gain")

# Certaines features ont gain=0 et sont absentes du dict → remises à 0
imp = np.array([scores.get(f, 0.0) for f in X.columns])

df_imp = pd.DataFrame({
    "feature":    X.columns,
    "importance": imp,
    "gain_pct":   imp / (imp.sum() + 1e-10) * 100,
}).sort_values("importance", ascending=False).reset_index(drop=True)

df_imp["gain_cumul_pct"] = df_imp["gain_pct"].cumsum()
df_imp["rank"] = df_imp.index + 1

# ─── Affichage TOP 50 ─────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("TOP 50 features XGBoost v2 (total_gain)")
print(f"{'=' * 80}")
print(f"{'Rank':>4}  {'Feature':<55}  {'Gain%':>6}  {'Cumul%':>7}")
print("-" * 80)
for _, row in df_imp.head(50).iterrows():
    print(f"{int(row['rank']):>4}  {row['feature']:<55}  {row['gain_pct']:>6.2f}  {row['gain_cumul_pct']:>7.2f}")

# ─── Seuils de coupure ────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("Nombre de features selon le seuil d'importance cumulée")
print(f"{'=' * 80}")
for thr in CUMULATIVE_THRESHOLDS:
    n_feats = int((df_imp["gain_cumul_pct"] <= thr * 100).sum()) + 1
    print(f"  {thr * 100:.0f}% du gain → {n_feats} features")
print(f"  Features avec importance = 0 : {int((df_imp['importance'] == 0).sum())}")

# ─── Sauvegarde ──────────────────────────────────────────────────────────────
out_csv = HERE / "feature_importance.csv"
df_imp.to_csv(out_csv, index=False)
print(f"\nClassement complet sauvegardé : {out_csv.name}")

# Liste des features couvrant 99.5% du gain cumulé
keep_mask        = df_imp["gain_cumul_pct"] <= 99.5
features_to_keep = df_imp[keep_mask]["feature"].tolist()

# Inclure la feature suivante si on n'est pas encore à 100%
if len(features_to_keep) < len(df_imp):
    features_to_keep.append(df_imp.iloc[len(features_to_keep)]["feature"])

out_txt = HERE / "features_to_keep.txt"
out_txt.write_text("\n".join(features_to_keep))
print(f"Liste des {len(features_to_keep)} features (≥99.5% gain) sauvegardée : {out_txt.name}")
