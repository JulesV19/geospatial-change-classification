"""
OOF (Out-of-Fold) Analysis — XGBoost v2
========================================
Strategy: StratifiedKFold 5 folds.
  Each fold trains on 80% of train, predicts on 20%.
  Predictions are assembled → full OOF on 100% of train.
  OOF macro F1 is the primary signal (correlates with private score).

Estimated runtime: ~N_ITERS × 5 folds × 1.9 it/s  (≈ 45 min for 3000 iters)

Outputs:
  oof_proba.npy              — OOF probabilities (N_train, 6)
  oof_preds.npy              — OOF predictions (N_train,)
  analyse_confusion.png      — normalised confusion matrix
  analyse_calibration.png    — max probability distribution per class
  analyse_erreurs_rares.csv  — misclassified examples (Industrial / Mega)
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
N_ITERS     = 3000
LR          = 0.05
ES_PATIENCE = 50   # rounds without improvement on val logloss before early stopping
MEGA_REPEAT       = 12    # reduced from 20: less memorisation
INDUSTRIAL_REPEAT = 8    # reduced from 10

TARGET_NAMES = ["Demolition", "Road", "Residential", "Commercial", "Industrial", "Mega Projects"]
SHORT_NAMES  = ["Demo", "Road", "Resi", "Comm", "Indu", "Mega"]
MANUAL_CW      = {0: 1.0, 1: 1.5, 2: 1.0, 3: 1.0, 4: 10.0, 5: 35.0}

t_global = time.time()

print("=" * 70)
print(f"OOF Analysis — {N_FOLDS}-Fold Stratified  ({N_ITERS} iterations / fold)")
print("=" * 70)

# ─── Load features ───────────────────────────────────────────────────────────
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
    print(f"\n  {X_all.shape[1]} features  (all)")

N = len(y_all)
print(f"  {N:,} total examples")
print(f"\n  Class distribution:")
for i, name in enumerate(TARGET_NAMES):
    n = (y_all == i).sum()
    print(f"    {name:<18}: {n:6,}  ({100*n/N:5.1f}%)")

print(f"\n  Mega repeat: ×{MEGA_REPEAT}   Industrial repeat: ×{INDUSTRIAL_REPEAT}  (built per fold, outside val)")

# ─── OOF Cross-validation ────────────────────────────────────────────────────
oof_proba = np.zeros((N, 6), dtype=np.float32)
oof_preds = np.full(N, -1, dtype=np.int32)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

f1_folds    = []
best_iters  = []   # best_iteration per fold (used for full retrain)
evals_results = [] # loss curves per fold

LOG_EVERY = 50  # print progress every N iterations


class _FoldLogger(xgb.callback.TrainingCallback):
    """Prints per-class F1 every LOG_EVERY iterations.
    Early stopping is handled natively by XGBoost on val logloss."""
    def __init__(self, period, t_start, X_val, y_val):
        self.period  = period
        self.t_start = t_start
        self.X_val   = X_val
        self.y_val   = y_val

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.period == 0:
            loss   = list(list(evals_log.values())[0].values())[0][-1]
            proba  = model.predict(xgb.DMatrix(self.X_val))
            preds  = proba.argmax(axis=1)
            f1_per = f1_score(self.y_val, preds, average=None,
                              labels=list(range(6)), zero_division=0)
            f1_mac = f1_per.mean()
            elapsed = time.time() - self.t_start
            m, s    = int(elapsed) // 60, int(elapsed) % 60
            print(
                f"  {epoch+1:>6}  {loss:>9.5f}  {f1_mac:>8.5f}"
                f"  {f1_per[4]:>6.4f}  {f1_per[5]:>6.4f}  {m:02d}:{s:02d}"
            )
        return False


for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
    t_fold = time.time()

    X_val = X_all.iloc[val_idx]
    y_val = y_all[val_idx]

    # Oversampling built on tr_idx only — no leakage into val_idx
    tr_y     = y_all[tr_idx]
    X_mega   = X_all.iloc[tr_idx][tr_y == 5].reset_index(drop=True)
    y_mega   = tr_y[tr_y == 5]
    X_indu   = X_all.iloc[tr_idx][tr_y == 4].reset_index(drop=True)
    y_indu   = tr_y[tr_y == 4]

    X_tr = pd.concat(
        [X_all.iloc[tr_idx]] + [X_mega] * MEGA_REPEAT + [X_indu] * INDUSTRIAL_REPEAT,
        ignore_index=True,
    )
    y_tr  = np.concatenate([y_all[tr_idx]] + [y_mega] * MEGA_REPEAT + [y_indu] * INDUSTRIAL_REPEAT)
    sw_tr = np.array([MANUAL_CW[c] for c in y_tr], dtype=float)

    n_mega_tr = (y_tr == 5).sum()
    n_indu_tr = (y_tr == 4).sum()
    print(f"\n{'─'*70}")
    print(f"  Fold {fold}/{N_FOLDS}  —  train: {len(tr_idx):,} + {n_mega_tr} Mega + {n_indu_tr} Indu  |  val: {len(val_idx):,}")
    print(f"  {'Iter':>6}  {'logloss':>9}  {'F1 macro':>8}  {'Indu':>6}  {'Mega':>6}  {'time':>5}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*5}")

    logger = _FoldLogger(LOG_EVERY, t_fold, X_val, y_val)

    model = xgb.XGBClassifier(
        objective             = "multi:softprob",
        num_class             = 6,
        eval_metric           = "mlogloss",
        tree_method           = "hist",
        device                = "cpu",
        learning_rate         = LR,
        n_estimators          = N_ITERS,
        max_depth             = 7,
        min_child_weight      = 6,
        gamma                 = 0.5,
        subsample             = 0.9,
        colsample_bytree      = 0.9,
        reg_alpha             = 0.0,
        reg_lambda            = 2.0,
        early_stopping_rounds = ES_PATIENCE,
        seed                  = SEED,
        verbosity             = 0,
        callbacks             = [logger],
    )
    model.fit(X_tr, y_tr, sample_weight=sw_tr,
              eval_set=[(X_val, y_val)], verbose=False)

    best_iter = model.best_iteration
    print(f"\n  Early stopping — best logloss @ iter {best_iter}")

    # Predictions at best logloss iter
    proba_val = model.predict_proba(X_val,
                    iteration_range=(0, best_iter + 1))
    preds_val = proba_val.argmax(axis=1)

    oof_proba[val_idx] = proba_val
    oof_preds[val_idx] = preds_val

    f1_fold = f1_score(y_val, preds_val, average="macro", zero_division=0)
    f1_folds.append(f1_fold)
    best_iters.append(best_iter)
    evals_results.append(model.evals_result())

    elapsed = time.time() - t_fold
    m, s   = int(elapsed) // 60, int(elapsed) % 60
    f1_per = f1_score(y_val, preds_val, average=None, zero_division=0)
    print(f"\n  Fold {fold} — best iter: {best_iter}  |  F1 macro: {f1_fold:.5f}  |  {m:02d}:{s:02d}")
    print(f"  F1 per class: " + "  ".join(
        f"{SHORT_NAMES[i]}={f1_per[i]:.3f}" for i in range(6)
    ))

# ─── Global OOF metrics ──────────────────────────────────────────────────────
f1_oof  = f1_score(y_all, oof_preds, average="macro", zero_division=0)
f1_std  = np.std(f1_folds)
ll_oof  = log_loss(y_all, oof_proba)

total_elapsed = time.time() - t_global
tm, ts = int(total_elapsed) // 60, int(total_elapsed) % 60

print(f"\n  OOF macro F1: {f1_oof:.5f}  ±{f1_std:.5f}  (inter-fold std)")
print(f"  OOF log-loss: {ll_oof:.5f}")
print(f"  Total time:   {tm:02d}:{ts:02d}")

# Save OOF
np.save(HERE / "oof_proba.npy", oof_proba)
np.save(HERE / "oof_preds.npy", oof_preds)
print(f"\n  OOF saved: oof_proba.npy  oof_preds.npy")

# ════════════════════════════════════════════════════════════════════════════
# 1. PER-CLASS METRICS
# ════════════════════════════════════════════════════════════════════════════
precision = precision_score(y_all, oof_preds, average=None, zero_division=0)
recall    = recall_score(y_all, oof_preds, average=None, zero_division=0)
f1_per    = f1_score(y_all, oof_preds, average=None, zero_division=0)
support   = np.bincount(y_all, minlength=6)
conf_max  = oof_proba.max(axis=1)

print("\n" + "━" * 70)
print("1. PER-CLASS METRICS  (full OOF)")
print("━" * 70)
hdr = f"  {'Class':<18} {'N':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}  {'Avg conf':>9}"
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
# 2. INTER-FOLD VARIANCE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 70)
print("2. INTER-FOLD STABILITY")
print("━" * 70)
print(f"  {'Fold':>5}  {'F1 macro':>10}  {'Δ avg':>8}")
print(f"  {'─'*5}  {'─'*10}  {'─'*8}")
for i, f1 in enumerate(f1_folds, 1):
    delta = f1 - f1_oof
    print(f"  {i:>5}  {f1:>10.5f}  {delta:>+8.5f}")
print(f"  {'Avg':>5}  {f1_oof:>10.5f}")
print(f"  {'Std':>5}  {f1_std:>10.5f}")
if f1_std > 0.005:
    print(f"\n  ⚠  High std ({f1_std:.4f}) — likely instability on rare classes.")

# ════════════════════════════════════════════════════════════════════════════
# 3. CONFUSION MATRIX
# ════════════════════════════════════════════════════════════════════════════
cm      = confusion_matrix(y_all, oof_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

print("\n" + "━" * 70)
print("3. CONFUSION MATRIX  (row = true, column = predicted)")
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
    ["Absolute", "Normalised (% of true label)"],
):
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(6)); ax.set_xticklabels(SHORT_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(SHORT_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(6):
        for j in range(6):
            val = data[i, j]
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax)
fig.suptitle(f"OOF Confusion Matrix — macro F1 = {f1_oof:.4f}", fontsize=13)
fig.tight_layout()
fig.savefig(HERE / "analyse_confusion.png", dpi=130)
print(f"\n  → analyse_confusion.png saved")

# ════════════════════════════════════════════════════════════════════════════
# 4. MOST FREQUENT CONFUSIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 70)
print("4. TOP 10 CONFUSIONS  (errors only)")
print("━" * 70)
errors = [
    (cm[i, j], 100 * cm_norm[i, j], TARGET_NAMES[i], TARGET_NAMES[j])
    for i in range(6) for j in range(6) if i != j and cm[i, j] > 0
]
errors.sort(reverse=True)
print(f"  {'True':<18} → {'Predicted':<18}  {'N':>7}  {'% of true':>10}")
print("  " + "─" * 60)
for n, pct, true, pred in errors[:10]:
    print(f"  {true:<18} → {pred:<18}  {n:>7,}  {pct:>9.2f}%")

# ════════════════════════════════════════════════════════════════════════════
# 5. PROBABILITY DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 70)
print("5. CONFIDENCE DISTRIBUTION  (max proba per class)")
print("━" * 70)
print(f"  {'Class':<18} {'N':>7}  {'< 0.5':>7}  {'0.5-0.8':>8}  {'> 0.8':>7}  {'Avg':>7}  {'Med':>7}")
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
    ax.set_xlabel("Confidence"); ax.set_ylabel("N")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig2.suptitle("OOF confidence distribution per class", fontsize=12)
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
# 7. COURBES D'APPRENTISSAGE PAR FOLD
# ════════════════════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(10, 5))
colors_fold = plt.cm.tab10(np.linspace(0, 0.5, N_FOLDS))
for i, (res, bi) in enumerate(zip(evals_results, best_iters)):
    loss = list(list(res.values())[0].values())[0]
    ax.plot(loss, color=colors_fold[i], alpha=0.8, linewidth=1.2,
            label=f"Fold {i+1}  (best={bi})")
    ax.axvline(bi, color=colors_fold[i], linestyle="--", linewidth=0.7, alpha=0.5)
ax.axvline(int(np.mean(best_iters)), color="black", linestyle="-", linewidth=1.5,
           label=f"Moy best iter = {int(np.mean(best_iters))}")
ax.set_xlabel("Itération"); ax.set_ylabel("mlogloss (val)")
ax.set_title("Courbes d'apprentissage — mlogloss par fold")
ax.legend(fontsize=8)
fig3.tight_layout()
fig3.savefig(HERE / "analyse_learning_curves.png", dpi=130)
print(f"\n  → analyse_learning_curves.png sauvegardé")

# ════════════════════════════════════════════════════════════════════════════
# 8. FULL RETRAIN + SUBMISSION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("FULL RETRAIN  (100% du train, n_iters = moyenne des best_iters OOF)")
print("═" * 70)

X_test   = feat["X_test"]
if keep_path.exists():
    X_test = X_test[[c for c in keep_cols if c in X_test.columns]]

n_final = int(np.median(best_iters))
# Même pipeline que la CV : duplication des rares sur tout le train (pas de val à protéger)
X_mega_full = X_all[y_all == 5].reset_index(drop=True)
y_mega_full = y_all[y_all == 5]
X_indu_full = X_all[y_all == 4].reset_index(drop=True)
y_indu_full = y_all[y_all == 4]

X_retrain = pd.concat(
    [X_all] + [X_mega_full] * MEGA_REPEAT + [X_indu_full] * INDUSTRIAL_REPEAT,
    ignore_index=True,
)
y_retrain = np.concatenate(
    [y_all] + [y_mega_full] * MEGA_REPEAT + [y_indu_full] * INDUSTRIAL_REPEAT
)
sw_retrain = np.array([MANUAL_CW[c] for c in y_retrain], dtype=float)
print(f"  n_estimators = {n_final}  (médiane des best_iters : {best_iters})")
print(f"  Retrain train size : {len(y_retrain):,} ({len(y_all):,} + duplication Mega×{MEGA_REPEAT} + Indu×{INDUSTRIAL_REPEAT})")

t_retrain = time.time()
final_model = xgb.XGBClassifier(
    objective        = "multi:softprob",
    num_class        = 6,
    eval_metric      = "mlogloss",
    tree_method      = "hist",
    device           = "cpu",
    learning_rate    = LR,
    n_estimators     = n_final,
    max_depth        = 7,
    min_child_weight = 6,
    gamma            = 0.5,
    subsample        = 0.9,
    colsample_bytree = 0.9,
    reg_alpha        = 0.0,
    reg_lambda       = 2.0,
    seed             = SEED,
    verbosity        = 0,
)
final_model.fit(X_retrain, y_retrain, sample_weight=sw_retrain, verbose=False)
print(f"  Retrain terminé en {time.time() - t_retrain:.1f}s")

test_proba = final_model.predict_proba(X_test)
test_preds = test_proba.argmax(axis=1)

np.save(HERE / "test_proba_xgboost_v2.npy", test_proba)

sub = pd.DataFrame({"Id": range(len(test_preds)), "change_type": test_preds})
sub_out = HERE / "submission_xgboost_v2.csv"
sub.to_csv(sub_out, index=False)

print(f"\n  Distribution des prédictions test :")
for cls, cnt in sub["change_type"].value_counts().items():
    print(f"    {cls:<18}: {cnt:6,}  ({100*cnt/len(sub):5.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ════════════════════════════════════════════════════════════════════════════
total_elapsed = time.time() - t_global
tm, ts = int(total_elapsed) // 60, int(total_elapsed) % 60

print("\n" + "═" * 70)
print("RÉSUMÉ FINAL")
print("═" * 70)
print(f"  F1 macro OOF   : {f1_oof:.5f}  ±{f1_std:.5f}")
print(f"  Log-Loss OOF   : {ll_oof:.5f}")
print(f"  Best iters OOF : {best_iters}  →  retrain avec {n_final}")
print(f"  Durée totale   : {tm:02d}:{ts:02d}")
print(f"\n  Classes problématiques (F1 < 0.90) :")
probs = [(f1_per[i], TARGET_NAMES[i]) for i in range(6) if f1_per[i] < 0.90]
probs.sort()
for f1_c, name in probs:
    print(f"    {name:<18}: F1 = {f1_c:.4f}")
print(f"\n  Fichiers générés :")
print(f"    oof_proba.npy  oof_preds.npy")
print(f"    analyse_confusion.png  analyse_calibration.png")
print(f"    analyse_learning_curves.png  analyse_erreurs_rares.csv")
print(f"    test_proba_xgboost_v2.npy  {sub_out.name}")
print()
