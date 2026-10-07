"""Train a linear head on frozen embeddings and write out-of-fold predictions.

For every backbone and every fold seed (columns fold_s<seed> of folds.csv):
    - for each fold k, fit on the other folds of the main cohort, predict fold k
    - standardization is fitted on the training part only (no leakage)
    - one logistic regression per lesion class (multi-label, one-vs-rest)
Rows outside the main cohort (fold -1) are never used for fitting. They get the
average prediction of the five fold models, kept for the separate exclude analysis.

Outputs:
    data/preds/oof_<backbone>_s<seed>.npz   probs (N, C) float32, fold (N,), classes
    results/heads_summary.txt               out-of-fold AUROC per class (aggregate numbers only)

Usage:
    python scripts/07_train_heads.py --backbones imagenet dinov2 --seeds 0 1 2
    python scripts/07_train_heads.py --backbones dinov2 --seeds 0 --C 0.1
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(str(text))


def parse_args():
    p = argparse.ArgumentParser(description="Train heads on frozen features, write OOF predictions.")
    p.add_argument("--manifest", default="data/interim/manifest.csv")
    p.add_argument("--folds", default="data/interim/folds.csv")
    p.add_argument("--classes", default="data/interim/classes.json")
    p.add_argument("--features_dir", default="data/features")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--backbones", nargs="+", default=["imagenet", "dinov2"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--C", type=float, default=0.1, help="inverse L2 strength of the logistic regression")
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--out_dir", default="data/preds")
    p.add_argument("--out_txt", default="results/heads_summary.txt")
    return p.parse_args()


def fit_predict_class(x_train, y_train, x_list, C, max_iter):
    """Fit one logistic regression, return predicted probabilities for each array in x_list."""
    if y_train.min() == y_train.max():  # only one class present in the training part
        const = float(y_train[0])
        return [np.full(len(x), const, dtype=np.float32) for x in x_list]
    model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return [model.predict_proba(x)[:, 1].astype(np.float32) for x in x_list]


def run_one(features, labels, fold, in_cohort, C, max_iter, n_jobs):
    n, n_classes = len(features), labels.shape[1]
    probs = np.zeros((n, n_classes), dtype=np.float32)
    outside = ~in_cohort
    outside_sum = np.zeros((int(outside.sum()), n_classes), dtype=np.float64)
    fold_ids = sorted(int(f) for f in np.unique(fold[in_cohort]))

    for k in fold_ids:
        train = in_cohort & (fold != k)
        test = in_cohort & (fold == k)
        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0) + 1e-6
        x_train = (features[train] - mean) / std
        x_test = (features[test] - mean) / std
        x_out = (features[outside] - mean) / std

        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_predict_class)(x_train, labels[train, c], [x_test, x_out], C, max_iter)
            for c in range(n_classes)
        )
        for c, (p_test, p_out) in enumerate(results):
            probs[test, c] = p_test
            outside_sum[:, c] += p_out

    probs[outside] = (outside_sum / len(fold_ids)).astype(np.float32)
    return probs


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    folds = pd.read_csv(args.folds)
    classes = json.loads(Path(args.classes).read_text())["classes"]

    if len(manifest) != len(folds):
        raise SystemExit(f"manifest has {len(manifest)} rows but folds.csv has {len(folds)}")
    if not (manifest["FileName"].astype(str).values == folds["FileName"].astype(str).values).all():
        raise SystemExit("FileName order differs between manifest and folds.csv; rerun 06")
    labels = manifest[classes].to_numpy(dtype=np.int8)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    say(f"rows: {len(manifest)}, classes: {len(classes)}, C: {args.C}")

    for backbone in args.backbones:
        path = Path(args.features_dir) / f"features_{backbone}_{args.size}.npy"
        features = np.load(path).astype(np.float32)
        if len(features) != len(manifest):
            raise SystemExit(f"{path} has {len(features)} rows, manifest has {len(manifest)}")
        if not np.isfinite(features).all():
            raise SystemExit(f"{path} contains NaN or inf")

        for seed in args.seeds:
            fold = folds[f"fold_s{seed}"].to_numpy()
            in_cohort = fold >= 0
            start = time.time()
            probs = run_one(features, labels, fold, in_cohort, args.C, args.max_iter, args.n_jobs)

            np.savez_compressed(
                Path(args.out_dir) / f"oof_{backbone}_s{seed}.npz",
                probs=probs, fold=fold.astype(np.int8), classes=np.array(classes),
            )

            say(f"\n== {backbone}, seed {seed} ({time.time() - start:.0f}s) ==")
            aucs = []
            for c, name in enumerate(classes):
                y, p = labels[in_cohort, c], probs[in_cohort, c]
                auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
                aucs.append(auc)
                say(f"  {name:28s} positives {int(y.sum()):5d}  AUROC {auc:.3f}")
            say(f"  macro AUROC: {np.nanmean(aucs):.3f}")

    Path(args.out_txt).write_text("\n".join(LINES) + "\n")
    say(f"\nwritten: {args.out_dir}/oof_*.npz, {args.out_txt}")


if __name__ == "__main__":
    main()