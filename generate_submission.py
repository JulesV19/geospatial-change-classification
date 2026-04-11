"""
Génération de la soumission — XGBoost v2 Full Retrain
======================================================
Principe :
  Réentraîne un modèle sur 100% du train. Chaque relance continue depuis
  le dernier checkpoint sauvegardé (reprise automatique).

  Premier lancement  : entraîne N_ITERS_START itérations depuis zéro.
  Relances suivantes : charge le dernier checkpoint et ajoute N_ITERS_EXTRA
                       itérations supplémentaires.

Sorties :
  submission_xgboost_v2.csv          — prédictions pour Kaggle (Id, change_type)
  test_proba_xgboost_v2.npy          — probabilités brutes
  cache/fullretrain/model_XXXXX.ubj  — checkpoint XGBoost (format binaire)
  cache/fullretrain/state.pkl        — nombre d'itérations actuelles

Prérequis :
  - cache/features.pkl   (lance feature_engineering.py d'abord)
  - features_to_keep.txt (optionnel, généré par feature_importance.py)
"""

import warnings
warnings.filterwarnings("ignore")

import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb

# ─── Config ──────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
CACHE_DIR    = HERE / "cache"

RETRAIN_DIR = CACHE_DIR / "fullretrain"
RETRAIN_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
MONITOR_PCT = 0.10    # fraction du train pour afficher la loss (pas d'early stopping)

# ── Nombre d'itérations ──────────────────────────────────────────────────────
# Premier lancement : n_iters_from_cv (voir plus bas).
# Relances suivantes : ajoute N_ITERS_EXTRA itérations au checkpoint existant.
N_ITERS_EXTRA = 2000  # ← ajuste selon le temps disponible

LEARNING_RATE = 0.05

TARGET_NAMES = ["Demolition", "Road", "Residential", "Commercial", "Industrial", "Mega Projects"]
MANUAL_CW    = {0: 1.3, 1: 2.8, 2: 0.27, 3: 0.40, 4: 30.0, 5: 265.0}

t0 = time.time()
print("=" * 70)
print("XGBoost v2 — Full Retrain (100% des données, reprise automatique)")
print("=" * 70)

# ─── Nombre d'itérations pour le premier lancement ───────────────────────────
n_iters_from_cv = 10000  # valeur par défaut (ajuste selon tes expériences)

# ─── Reprise depuis un checkpoint existant ? ──────────────────────────────────
state_path = RETRAIN_DIR / "state.pkl"

if state_path.exists():
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    current_iters = state["iters"]
    model_path    = state["model_path"]

    if not Path(model_path).exists():
        print(f"\n  ⚠  Checkpoint {model_path} introuvable — reprise depuis zéro.")
        current_iters = 0
        model_path    = None
    else:
        n_new_iters = N_ITERS_EXTRA
        print(f"\n  Checkpoint existant : {Path(model_path).name}  ({current_iters} itérations)")
        print(f"  → Ajout de {n_new_iters} itérations  (total visé : {current_iters + n_new_iters})")
else:
    current_iters = 0
    model_path    = None
    n_new_iters   = n_iters_from_cv
    print(f"\n  Aucun checkpoint — entraînement depuis zéro ({n_new_iters} itérations)")

# ─── Chargement des features ──────────────────────────────────────────────────
feat_cache = CACHE_DIR / "features.pkl"
if not feat_cache.exists():
    raise FileNotFoundError(f"{feat_cache} introuvable. Lance feature_engineering.py d'abord.")

print("\nChargement des features...")
with open(feat_cache, "rb") as f:
    feat_ckpt = pickle.load(f)

X_train = feat_ckpt["X_train"]
X_test  = feat_ckpt["X_test"]
y       = feat_ckpt["y"]
print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

# ─── Filtrage des features ────────────────────────────────────────────────────
keep_path = HERE / "features_to_keep.txt"
if keep_path.exists():
    keep_cols = [c.strip() for c in keep_path.read_text().splitlines() if c.strip()]
    keep_cols = [c for c in keep_cols if c in X_train.columns]
    X_train = X_train[keep_cols]
    X_test  = X_test[keep_cols]
    print(f"  Filtrage : {len(keep_cols)} features (source: {keep_path.name})")
else:
    print("  (features_to_keep.txt absent — toutes les features utilisées)")

# ─── Poids d'entraînement ─────────────────────────────────────────────────────
sw_train = np.array([MANUAL_CW[c] for c in y], dtype=float)

print(f"\nDistribution des classes :")
for i in range(6):
    print(f"  {TARGET_NAMES[i]:15s}: {(y == i).sum():6d} ({100 * (y == i).mean():5.1f}%)")

# ─── Monitoring (sous-échantillon pour afficher la loss) ──────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=MONITOR_PCT, random_state=SEED)
_, monitor_idx = next(sss.split(X_train, y))
X_monitor = X_train.iloc[monitor_idx]
y_monitor  = y[monitor_idx]
print(f"\n  Monitoring loss sur {len(X_monitor)} exemples ({MONITOR_PCT * 100:.0f}% du train)")

# ─── Paramètres XGBoost ───────────────────────────────────────────────────────
default_tunable = dict(
    max_depth        = 10,
    min_child_weight = 3,
    subsample        = 0.9,
    colsample_bytree = 0.9,
    reg_alpha        = 0.0,
    reg_lambda       = 1.0,
    gamma            = 0.0,
)

optuna_path = CACHE_DIR / "optuna_xgboost_best_params.pkl"
if optuna_path.exists():
    with open(optuna_path, "rb") as f:
        optuna_data = pickle.load(f)
    tunable = optuna_data["best_params"]
    print(f"\n  Paramètres Optuna chargés (Macro F1 CV = {optuna_data['best_score_cv']:.5f})")
else:
    tunable = default_tunable
    print("\n  Paramètres Optuna non trouvés — valeurs par défaut utilisées")

xgb_params = dict(
    objective     = "multi:softprob",
    num_class     = 6,
    eval_metric   = "mlogloss",
    tree_method   = "hist",
    device        = "cpu",
    learning_rate = LEARNING_RATE,
    n_estimators  = n_new_iters,
    seed          = SEED,
    verbosity     = 0,
    **tunable,
)

total_iters = current_iters + n_new_iters
print(f"\n  Entraînement : {n_new_iters} itérations  (total après : {total_iters})")
print(f"  Learning rate : {LEARNING_RATE}")
print(f"  F1 macro affiché toutes les 200 itérations\n")

F1_PRINT_FREQ   = 200
SUBMISSION_FREQ = 500

SUBMISSIONS_DIR = HERE / "submissions_iter"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# ─── Callback : suivi détaillé toutes les F1_PRINT_FREQ iters ────────────────
# ─── et sauvegarde une soumission CSV toutes les SUBMISSION_FREQ iters       ───
class F1MonitorCallback(xgb.callback.TrainingCallback):
    def __init__(self, X_val, y_val, X_test, print_freq, iters_offset, n_total_new):
        self.X_val          = X_val
        self.y_val          = y_val
        self.X_test         = X_test
        self.print_freq     = print_freq
        self.iters_offset   = iters_offset
        self.n_total_new    = n_total_new
        self.t0             = time.time()
        self._header_done   = False

    def _fmt_time(self, seconds):
        """Formate un nombre de secondes en mm:ss ou hh:mm."""
        s = int(seconds)
        if s < 3600:
            return f"{s // 60:02d}:{s % 60:02d}"
        return f"{s // 3600:d}h{(s % 3600) // 60:02d}"

    def after_iteration(self, model, epoch, evals_log):
        it    = epoch + 1
        total = self.iters_offset + it

        # ── Suivi toutes les F1_PRINT_FREQ iters ─────────────────────────────
        if it % self.print_freq == 0:
            elapsed = time.time() - self.t0
            speed   = it / elapsed if elapsed > 0 else 0
            eta_s   = (self.n_total_new - it) / speed if speed > 0 else 0
            pct     = 100.0 * it / self.n_total_new

            loss_val = float("nan")
            for ds_vals in evals_log.values():
                if "mlogloss" in ds_vals:
                    loss_val = ds_vals["mlogloss"][-1]
                    break

            proba   = model.predict(xgb.DMatrix(self.X_val))
            preds   = proba.argmax(axis=1)
            f1_per  = f1_score(self.y_val, preds, average=None,
                               labels=list(range(6)), zero_division=0)
            f1_mac  = f1_per.mean()

            if not self._header_done:
                print(
                    f"\n  {'Iter':>6}  {'Progr':>6}  {'Elapsed':>7}  {'ETA':>5}  "
                    f"{'it/s':>5}  {'LogLoss':>9}  {'F1_mac':>7}  {'Indus':>6}  {'Mega':>6}"
                )
                print(f"  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*6}")
                self._header_done = True

            print(
                f"  {total:6d}  {pct:5.1f}%  {self._fmt_time(elapsed):>7}  "
                f"{self._fmt_time(eta_s):>5}  {speed:5.1f}  "
                f"{loss_val:9.5f}  {f1_mac:7.5f}  {f1_per[4]:6.4f}  {f1_per[5]:6.4f}"
            )

        # ── Soumission CSV toutes les SUBMISSION_FREQ iters ──────────────────
        if total % SUBMISSION_FREQ == 0:
            test_proba = model.predict(xgb.DMatrix(self.X_test))
            test_preds = test_proba.argmax(axis=1)
            sub     = pd.DataFrame({"Id": range(len(test_preds)), "change_type": test_preds})
            sub_out = SUBMISSIONS_DIR / f"submission_iter{total:06d}.csv"
            sub.to_csv(sub_out, index=False)
            print(f"  → Soumission sauvegardée : {sub_out.name}")

        return False


# ─── Entraînement (reprise depuis checkpoint si disponible) ───────────────────
f1_cb = F1MonitorCallback(
    X_val        = X_monitor,
    y_val        = y_monitor,
    X_test       = X_test,
    print_freq   = F1_PRINT_FREQ,
    iters_offset = current_iters,
    n_total_new  = n_new_iters,
)

model = xgb.XGBClassifier(**xgb_params, callbacks=[f1_cb])

model.fit(
    X_train, y,
    sample_weight = sw_train,
    eval_set      = [(X_monitor, y_monitor)],
    verbose       = False,   # désactivé — remplacé par le callback F1
    xgb_model     = model_path,
)

print(f"\n  Entraînement terminé en {time.time() - t0:.1f}s")

# ─── Sauvegarde du checkpoint ─────────────────────────────────────────────────
new_model_path = RETRAIN_DIR / f"model_{total_iters:06d}.ubj"
model.save_model(str(new_model_path))

# Supprime l'ancien checkpoint pour économiser l'espace
if model_path and Path(model_path).exists():
    Path(model_path).unlink()

with open(state_path, "wb") as f:
    pickle.dump({"iters": total_iters, "model_path": str(new_model_path)}, f)

print(f"  Checkpoint sauvegardé : {new_model_path.name}  ({total_iters} itérations au total)")

# ─── Prédictions sur le test ──────────────────────────────────────────────────
print("\nPrédiction sur le jeu de test...")
test_proba = model.predict_proba(X_test)
test_preds = test_proba.argmax(axis=1)

# ─── Sauvegarde des probabilités brutes ──────────────────────────────────────
proba_out = HERE / "test_proba_xgboost_v2.npy"
np.save(proba_out, test_proba)
print(f"  Probabilités sauvegardées : {proba_out.name}")

# ─── Génération de la soumission ──────────────────────────────────────────────
sub     = pd.DataFrame({"Id": range(len(test_preds)), "change_type": test_preds})
sub_out = HERE / "submission_xgboost_v2.csv"
sub.to_csv(sub_out, index=False)

print(f"\n{'=' * 70}")
print(f"SOUMISSION : {sub_out.name}  ({total_iters} itérations)")
print(f"{'=' * 70}")
dist = pd.Series(test_preds).value_counts().sort_index()
for cls, count in dist.items():
    print(f"  {TARGET_NAMES[cls]:15s}: {count:6d} ({100 * count / len(test_preds):5.1f}%)")

print(f"\n  Temps total : {time.time() - t0:.1f}s")
print(f"\n  Pour continuer l'entraînement : relance ce script.")
print(f"  Chaque relance ajoute {N_ITERS_EXTRA} itérations (modifie N_ITERS_EXTRA si besoin).")
