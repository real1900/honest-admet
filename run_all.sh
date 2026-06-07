#!/usr/bin/env bash
# Reproduce Honest ADMET end-to-end. Run from the repo root.
#   uv venv --python 3.11 .venv && uv pip install -e ".[deep]"
#   bash run_all.sh
# Set PY to point at your interpreter (default: the project venv).
set -euo pipefail
PY=${PY:-.venv/bin/python}

echo "[1/11] environment + live TDC access"; $PY experiments/00_smoke_test.py
echo "[2/11] split diagnostics + leakage gradient"; $PY experiments/01_validate_splits.py
echo "[3/11] leaderboard reproduction (official split)"; $PY experiments/03_leaderboard_repro.py --datasets all --models xgboost
echo "[4/11] gap atlas (22 endpoints, matched-replicate stats)"; \
  $PY experiments/02_baselines.py --datasets all --splits random,scaffold,cluster \
      --split-seeds 8 --model-seeds 3 --models xgboost --out results/atlas_full.csv
echo "[5/11] applicability-domain mechanism"; $PY experiments/06_applicability_domain.py
echo "[6/11] uncertainty under shift (conformal + selective prediction)"; \
  $PY experiments/04_uncertainty.py --datasets all --model xgboost --split-seeds 5 --members 5 \
      --out results/uncertainty.csv
echo "[7/11] unifying analysis (gap vs abstention benefit)"; $PY experiments/07_unifying.py
# Foundation-model arm: extract embeddings in a torch-only process FIRST (avoids the
# macOS torch+XGBoost OpenMP segfault), then run the head on the warm cache.
echo "[8/11] cache ChemBERTa embeddings (torch only)"; $PY experiments/09_embed_cache.py --fm chemberta
echo "[9/11] foundation-model gap (XGBoost head on cached embeddings)"; \
  KMP_DUPLICATE_LIB_OK=TRUE $PY experiments/08_foundation.py --datasets all --fm chemberta \
      --split-seeds 5 --model-seeds 3
echo "[10/11] review-fix stats (relative medians, block-bootstrap CIs)"; $PY experiments/12_review_fixes.py
echo "[11/11] tables + figures"; $PY experiments/11_tables.py && $PY experiments/10_figures.py

echo
echo "Done. Results in results/, figures in paper/figures/, manuscript in paper/honest_admet.tex"
echo "Conformal correctness self-test: $PY experiments/05_conformal_selftest.py"
