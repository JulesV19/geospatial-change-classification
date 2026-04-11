"""
Analyse OOF (Out-of-Fold) — XGBoost v2
========================================
Stratégie : StratifiedKFold 5 folds.
  Chaque fold entraîne sur 80 % du train, prédit sur 20 %.
  Les prédictions sont assemblées → OOF complet sur 100 % du train.
  Le F1 macro OOF est la boussole principale (corrèle avec le private score).

Durée estimée : ~N_ITERS × 5 folds × 1.9 it/s  (≈ 45 min pour 3000 iters)

Sorties :
  oof_proba.npy              — probabilités OOF (N_train, 6)
  oof_preds.npy              — prédictions OOF (N_train,)
  analyse_confusion.png      — matrice de confusion normalisée
  analyse_calibration.png    — distribution des probabilités max par classe
  analyse_erreurs_rares.csv  — exemples mal classifiés (Industrial / Mega)
"""

import warnings
warnings.filterwarnings("ignore")

import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, log_loss,
)
import xgboost as xgb

# ─── Config ──────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
CACHE_DIR = HERE / "cache"
SEED      = 42
N_FOLDS   = 5
N_ITERS   = 3000   # suffisant (plateau observé vers iter 3000-4000)
LR        = 0.05

TARGET_NAMES = ["Demolition", "Road", "Residential", "Commercial", "Industrial", "Mega Projects"]
SHORT_NAMES  = ["Demo", "Road", "Resi", "Comm", "Indu", "Mega"]
MANUAL_CW    = {0: 1.3, 1: 2.8, 2: 0.27, 3: 0.40, 4: 30.0, 5: 265.0}

t_global = time.time()

print("=" * 70)
print(f"Analyse OOF — {N_FOLDS}-Fold Stratifié  ({N_ITERS} itérations / fold)")
print("=" * 70)

# ─── Chargement features ─────────────────────────────────────────────────────
feat_cache = CACHE_DIR / "features.pkl"
with open(feat_cache, "rb") as f:
    feat = pickle.load(f)

X_all = feat["X_train"]
y_all = feat["y"]

keep_path = HERE / "features_to_keep.txt"
if keep_path.exists():
    keep_cols = [c.strip() for c in keep_path.read_text().splitlines() if c.strip()]
    keep_cols = [c for c in keep_cols if c in X_all.columns]
    X_all = X_all[keep_cols]
    print(f"\n  {len(keep_cols)} features  (features_to_keep.txt)")
else:
    print(f"\n  {X_all.shape[1]} features  (toutes)")

N = len(y_all)
print(f"  {N:,} exemples au total")
print(f"\n  Distribution des classes :")
for i, name in enumerate(TARGET_NAMES):
    n = (y_all == i).sum()
    print(f"    {name:<18}: {n:6,}  ({100*n/N:5.1f}%)")

# ─── Cross-validation OOF ────────────────────────────────────────────────────
oof_proba = np.zeros((N, 6), dtype=np.float32)
oof_preds = np.full(N, -1, dtype=np.int32)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

f1_folds = []

print(f"\n{'─'*70}")
print(f"  {'Fold':>5}  {'Train':>8}  {'Val':>8}  {'F1 macro':>10}  {'Elapsed':>9}")
print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*9}")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
    t_fold = time.time()

    X_tr  = X_all.iloc[tr_idx]
    X_val = X_all.iloc[val_idx]
    y_tr  = y_all[tr_idx]
    y_val = y_all[val_idx]
    sw_tr = np.array([MANUAL_CW[c] for c in y_tr], dtype=float)

    model = xgb.XGBClassifier(
        objective        = "multi:softprob",
        num_class        = 6,
        eval_metric      = "mlogloss",
        tree_method      = "hist",
        device           = "cpu",
        learning_rate    = LR,
        n_estimators     = N_ITERS,
        max_depth        = 10,
        min_child_weight = 3,
        subsample        = 0.9,
        colsample_bytree = 0.9,
        reg_alpha        = 0.0,
        reg_lambda       = 1.0,
        seed             = SEED,
        verbosity        = 0,
    )
    model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)

    proba_val = model.predict_proba(X_val)
    preds_val = proba_val.argmax(axis=1)

    oof_proba[val_idx] = proba_val
    oof_preds[val_idx] = preds_val

    f1_fold = f1_score(y_val, preds_val, average="macro", zero_division=0)
    f1_folds.append(f1_fold)

    elapsed = time.time() - t_fold
    m, s = int(elapsed) // 60, int(elapsed) % 60
    print(f"  {fold:>5}  {len(tr_idx):>8,}  {len(val_idx):>8,}  {f1_fold:>10.5f}  {m:02d}:{s:02d}")

# ─── Métriques OOF globales ──────────────────────────────────────────────────
f1_oof  = f1_score(y_all, oof_preds, average="macro", zero_division=0)
f1_std  = np.std(f1_folds)
ll_oof  = log_loss(y_all, oof_proba)

total_elapsed = time.time() - t_global
tm, ts = int(total_elapsed) // 60, int(total_elapsed) % 60

print(f"\n  F1 macro OOF : {f1_oof:.5f}  ±{f1_std:.5f}  (std inter-folds)")
print(f"  Log-Loss OOF : {ll_oof:.5f}")
print(f"  Durée totale : {tm:02d}:{ts:02d}")

# Sauvegarde OOF
np.save(HERE / "oof_proba.npy", oof_proba)
np.save(HERE / "oof_preds.npy", oof_preds)
print(f"\n  OOF sauvegardé : oof_proba.npy  oof_preds.npy")

# ════════════════════════════════════════════════════════════════════════════
# 1. MÉTRIQUES PAR CLASSE
# ════════════════════════════════════════════════════════════════════════════
precision = precision_score(y_all, oof_preds, average=None, zero_division=0)
recall    = recall_score(y_all, oof_preds, average=None, zero_division=0)
f1_per    = f1_score(y_all, oof_preds, average=None, zero_division=0)
support   = np.bincount(y_all, minlength=6)
conf_max  = oof_proba.max(axis=1)

print("\n" + "━" * 70)
print("1. MÉTRIQUES PAR CLASSE  (OOF complet)")
print("━" * 70)
hdr = f"  {'Classe':<18} {'N':>8} {'Précision':>10} {'Rappel':>8} {'F1':>8}  {'Conf moy':>9}"
print(hdr)
print("  " + "─" * (len(hdr) - 2))
for i, name in enumerate(TARGET_NAMES):
    mask_i = y_all == i
    conf_i = conf_max[mask_i].mean() if mask_i.sum() > 0 else float("nan")
    flag   = "  ← !" if f1_per[i] < 0.80 else ""
    print(
        f"  {name:<18} {support[i]:>8,} {precision[i]:>10.4f}"
        f" {recall[i]:>8.4f} {f1_per[i]:>8.4f}  {conf_i:>9.4f}{flag}"
    )

# ════════════════════════════════════════════════════════════════════════════
# 2. VARIANCE INTER-FOLDS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 70)
print("2. STABILITÉ INTER-FOLDS")
print("━" * 70)
print(f"  {'Fold':>5}  {'F1 macro':>10}  {'Δ moy':>8}")
print(f"  {'─'*5}  {'─'*10}  {'─'*8}")
for i, f1 in enumerate(f1_folds, 1):
    delta = f1 - f1_oof
    print(f"  {i:>5}  {f1:>10.5f}  {delta:>+8.5f}")
print(f"  {'Moy':>5}  {f1_oof:>10.5f}")
print(f"  {'Std':>5}  {f1_std:>10.5f}")
if f1_std > 0.005:
    print(f"\n  ⚠  Std élevée ({f1_std:.4f}) — instabilité sur les classes rares probable.")

# ════════════════════════════════════════════════════════════════════════════
# 3. MATRICE DE CONFUSION
# ════════════════════════════════════════════════════════════════════════════
cm      = confusion_matrix(y_all, oof_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

print("\n" + "━" * 70)
print("3. MATRICE DE CONFUSION  (ligne = vrai, colonne = prédit)")
print("━" * 70)
col_w = 8
print("  " + " " * 6 + "".join(f"{s:>{col_w}}" for s in SHORT_NAMES))
for i, name in enumerate(SHORT_NAMES):
    row = "  " + f"{name:<6}" + "".join(f"{cm[i, j]:>{col_w},}" for j in range(6))
    print(row)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, data, fmt, title in zip(
    axes,
    [cm, cm_norm],
    [",", ".2f"],
    ["Absolue", "Normalisée (% du vrai label)"],
):
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(6)); ax.set_xticklabels(SHORT_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(SHORT_NAMES)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
    ax.set_title(title)
    for i in range(6):
        for j in range(6):
            val = data[i, j]
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax)
fig.suptitle(f"Matrice de confusion OOF — F1 macro = {f1_oof:.4f}", fontsize=13)
fig.tight_layout()
fig.savefig(HERE / "analyse_confusion.png", dpi=130)
print(f"\n  → analyse_confusion.png sauvegardé")

# ════════════════════════════════════════════════════════════════════════════
# 4. CONFUSIONS LES PLUS FRÉQUENTES
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 70)
print("4. TOP 10 CONFUSIONS  (erreurs uniquement)")
print("━" * 70)
errors = [
    (cm[i, j], 100 * cm_norm[i, j], TARGET_NAMES[i], TARGET_NAMES[j])
    for i in range(6) for j in range(6) if i != j and cm[i, j] > 0
]
errors.sort(reverse=True)
print(f"  {'Vrai':<18} → {'Prédit':<18}  {'N':>7}  {'% du vrai':>10}")
print("  " + "─" * 60)
for n, pct, true, pred in errors[:10]:
    print(f"  {true:<18} → {pred:<18}  {n:>7,}  {pct:>9.2f}%")

# ════════════════════════════════════════════════════════════════════════════
# 5. DISTRIBUTION DES PROBABILITÉS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 70)
print("5. DISTRIBUTION DE LA CONFIANCE  (proba max par classe)")
print("━" * 70)
print(f"  {'Classe':<18} {'N':>7}  {'< 0.5':>7}  {'0.5-0.8':>8}  {'> 0.8':>7}  {'Moy':>7}  {'Méd':>7}")
print("  " + "─" * 70)
for i, name in enumerate(TARGET_NAMES):
    mask  = y_all == i
    c_cls = conf_max[mask]
    if len(c_cls) == 0:
        continue
    low  = (c_cls < 0.5).sum()
    mid  = ((c_cls >= 0.5) & (c_cls < 0.8)).sum()
    high = (c_cls >= 0.8).sum()
    print(
        f"  {name:<18} {mask.sum():>7,}  {low:>7,}  {mid:>8,}  {high:>7,}"
        f"  {c_cls.mean():>7.4f}  {np.median(c_cls):>7.4f}"
    )

fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
for i, (ax, name) in enumerate(zip(axes2.flat, TARGET_NAMES)):
    mask   = y_all == i
    c_tp   = conf_max[mask & (oof_preds == i)]
    c_fn   = conf_max[mask & (oof_preds != i)]
    c_fp   = conf_max[(y_all != i) & (oof_preds == i)]
    bins   = np.linspace(0, 1, 26)
    ax.hist(c_tp, bins=bins, alpha=0.7, label=f"TP ({len(c_tp)})", color="steelblue")
    ax.hist(c_fn, bins=bins, alpha=0.7, label=f"FN ({len(c_fn)})", color="tomato")
    ax.hist(c_fp, bins=bins, alpha=0.5, label=f"FP ({len(c_fp)})", color="orange")
    ax.set_title(f"{name}  (F1={f1_per[i]:.3f})", fontsize=9)
    ax.set_xlabel("Confiance"); ax.set_ylabel("N")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig2.suptitle("Distribution de la confiance OOF par classe", fontsize=12)
fig2.tight_layout()
fig2.savefig(HERE / "analyse_calibration.png", dpi=130)
print(f"\n  → analyse_calibration.png sauvegardé")

# ════════════════════════════════════════════════════════════════════════════
# 6. CLASSES RARES — ANALYSE DÉTAILLÉE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 70)
print("6. CLASSES RARES — Industrial & Mega Projects")
print("━" * 70)

all_df = X_all.copy()
all_df["_true"]    = y_all
all_df["_pred"]    = oof_preds
all_df["_conf"]    = conf_max
all_df["_proba_4"] = oof_proba[:, 4]
all_df["_proba_5"] = oof_proba[:, 5]

for cls_idx, cls_name, proba_col in [
    (4, "Industrial",    "_proba_4"),
    (5, "Mega Projects", "_proba_5"),
]:
    mask_true = all_df["_true"] == cls_idx
    n_true    = mask_true.sum()

    tp = ((all_df["_true"] == cls_idx) & (all_df["_pred"] == cls_idx)).sum()
    fn = ((all_df["_true"] == cls_idx) & (all_df["_pred"] != cls_idx)).sum()
    fp = ((all_df["_true"] != cls_idx) & (all_df["_pred"] == cls_idx)).sum()

    print(f"\n  {cls_name}  (classe {cls_idx})")
    print(f"    Total : {n_true}    TP : {tp} ({100*tp/n_true:.1f}%)    FN : {fn}    FP : {fp}")

    fn_df = all_df[mask_true & (all_df["_pred"] != cls_idx)]
    if len(fn_df) > 0:
        print(f"    FN prédits comme :")
        for pred_cls, cnt in fn_df["_pred"].value_counts().items():
            conf_m = fn_df[fn_df["_pred"] == pred_cls]["_conf"].mean()
            print(f"      → {TARGET_NAMES[pred_cls]:<18}: {cnt:4d}  (conf moy = {conf_m:.3f})")

    p_true = all_df.loc[mask_true, proba_col]
    print(f"    Proba '{cls_name}' sur vrais exemples : "
          f"min={p_true.min():.4f}  méd={p_true.median():.4f}  max={p_true.max():.4f}")

rare_errors = all_df[
    (all_df["_true"].isin([4, 5]) | all_df["_pred"].isin([4, 5])) &
    (all_df["_true"] != all_df["_pred"])
].copy()
rare_errors["_true_name"] = rare_errors["_true"].map(lambda x: TARGET_NAMES[x])
rare_errors["_pred_name"] = rare_errors["_pred"].map(lambda x: TARGET_NAMES[x])
rare_errors.to_csv(HERE / "analyse_erreurs_rares.csv", index=False)
print(f"\n  → analyse_erreurs_rares.csv  ({len(rare_errors)} lignes)")

# ════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ════════════════════════════════════════════════════════════════════════════
total_elapsed = time.time() - t_global
tm, ts = int(total_elapsed) // 60, int(total_elapsed) % 60

print("\n" + "═" * 70)
print("RÉSUMÉ")
print("═" * 70)
print(f"  F1 macro OOF   : {f1_oof:.5f}  ±{f1_std:.5f}")
print(f"  Log-Loss OOF   : {ll_oof:.5f}")
print(f"  Durée totale   : {tm:02d}:{ts:02d}")
print(f"\n  Classes problématiques (F1 < 0.90) :")
probs = [(f1_per[i], TARGET_NAMES[i]) for i in range(6) if f1_per[i] < 0.90]
probs.sort()
for f1_c, name in probs:
    print(f"    {name:<18}: F1 = {f1_c:.4f}")
print(f"\n  Fichiers générés :")
print(f"    oof_proba.npy  oof_preds.npy")
print(f"    analyse_confusion.png  analyse_calibration.png")
print(f"    analyse_erreurs_rares.csv")
print()
