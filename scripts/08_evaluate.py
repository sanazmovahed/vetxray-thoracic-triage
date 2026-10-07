"""Evaluate out-of-fold predictions: discrimination, calibration, selective prediction.

Reads the predictions written by 07 and writes aggregate numbers only.

Sections per backbone:
  1. Discrimination: AUROC and average precision per class, mean +- sd over fold seeds,
     and a 95% cluster-bootstrap interval (whole groups are resampled, so images of one
     animal are never treated as independent).
  2. Calibration: ECE and Brier score per class, raw and after Platt scaling that is
     cross-fitted over the folds of seed 0.
  3. Selective prediction: per class, the images closest to the decision threshold are set
     aside and balanced accuracy is computed on the rest; compared with setting aside random images.
  4. Results per X-ray manufacturer.
  5. Excluded images: is the model more uncertain on them than on the main cohort?

Usage:
    python scripts/08_evaluate.py
    python scripts/08_evaluate.py --backbones dinov2 --n_boot 200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

LINES = []
EPS = 1e-6


def say(text=""):
    print(text, flush=True)
    LINES.append(str(text))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate out-of-fold predictions.")
    p.add_argument("--manifest", default="data/interim/manifest.csv")
    p.add_argument("--classes", default="data/interim/classes.json")
    p.add_argument("--preds_dir", default="data/preds")
    p.add_argument("--backbones", nargs="+", default=["imagenet", "dinov2"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--n_bins", type=int, default=15)
    p.add_argument("--coverages", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7, 0.5])
    p.add_argument("--min_manufacturer", type=int, default=300)
    p.add_argument("--out_txt", default="results/eval_summary.txt")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--rng_seed", type=int, default=0)
    return p.parse_args()


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def safe_auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def safe_ap(y, p):
    return average_precision_score(y, p) if y.sum() > 0 else np.nan


def ece(y, p, n_bins):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return total


def entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def platt_cross_fit(y, p, fold):
    """Platt scaling per class; each fold is recalibrated by a map fitted on the other folds."""
    out = np.zeros_like(p)
    for c in range(p.shape[1]):
        for k in np.unique(fold):
            train, test = fold != k, fold == k
            if y[train, c].min() == y[train, c].max():
                out[test, c] = p[test, c]
                continue
            lr = LogisticRegression(C=1e4, max_iter=1000)
            lr.fit(logit(p[train, c])[:, None], y[train, c])
            out[test, c] = lr.predict_proba(logit(p[test, c])[:, None])[:, 1]
    return out


def macro_auc(y, p, min_pos=5, min_neg=5):
    vals = []
    for c in range(y.shape[1]):
        pos = y[:, c].sum()
        if pos >= min_pos and len(y) - pos >= min_neg:
            vals.append(roc_auc_score(y[:, c], p[:, c]))
    return float(np.mean(vals)) if vals else np.nan


def cluster_bootstrap(y, p, groups, n_boot, rng):
    """95% interval of per-class AUROC and macro AUROC, resampling whole groups."""
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    n_classes = y.shape[1]
    aucs = np.full((n_boot, n_classes), np.nan)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        for c in range(n_classes):
            aucs[b, c] = safe_auc(y[idx, c], p[idx, c])
    macro = np.nanmean(aucs, axis=1)
    lo, hi = np.nanpercentile(aucs, [2.5, 97.5], axis=0)
    return lo, hi, np.nanpercentile(macro, [2.5, 97.5])


def main():
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)
    manifest = pd.read_csv(args.manifest)
    classes = json.loads(Path(args.classes).read_text())["classes"]
    labels = manifest[classes].to_numpy(dtype=np.int8)
    groups_all = manifest["group"].to_numpy()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for backbone in args.backbones:
        runs = []
        for seed in args.seeds:
            path = Path(args.preds_dir) / f"oof_{backbone}_s{seed}.npz"
            runs.append(np.load(path))
        fold0 = runs[0]["fold"]
        in_cohort = fold0 >= 0
        if len(fold0) != len(manifest):
            raise SystemExit("predictions and manifest have different lengths; rerun 06 and 07")
        y = labels[in_cohort]
        groups = groups_all[in_cohort]
        probs_seeds = [r["probs"][in_cohort] for r in runs]
        pbar = np.mean(probs_seeds, axis=0)

        say(f"\n{'=' * 8} {backbone} {'=' * 8}")
        say(f"cohort images: {len(y)}, groups: {len(np.unique(groups))}, seeds: {args.seeds}")

        # 1. discrimination
        say("\n-- 1. Discrimination (OOF, seed-averaged probabilities) --")
        lo, hi, macro_ci = cluster_bootstrap(y, pbar, groups, args.n_boot, rng)
        rows = []
        for c, name in enumerate(classes):
            per_seed = [safe_auc(y[:, c], p[:, c]) for p in probs_seeds]
            rows.append({
                "class": name, "positives": int(y[:, c].sum()),
                "auroc": safe_auc(y[:, c], pbar[:, c]), "auroc_lo": lo[c], "auroc_hi": hi[c],
                "auroc_seed_sd": float(np.std(per_seed)),
                "ap": safe_ap(y[:, c], pbar[:, c]), "prevalence": float(y[:, c].mean()),
            })
        table = pd.DataFrame(rows)
        say(table.round(3).to_string(index=False))
        macro = float(table["auroc"].mean())
        say(f"macro AUROC {macro:.3f}  (95% CI {macro_ci[0]:.3f}-{macro_ci[1]:.3f}); "
            f"per-seed macro: {[round(macro_auc(y, p, 1, 1), 3) for p in probs_seeds]}")

        # 2. calibration
        say("\n-- 2. Calibration (ECE, Brier; Platt cross-fitted over seed-0 folds) --")
        fold_c = fold0[in_cohort]
        p_cal = platt_cross_fit(y, pbar, fold_c)
        cal_rows = []
        for c, name in enumerate(classes):
            cal_rows.append({
                "class": name,
                "ece_raw": ece(y[:, c], pbar[:, c], args.n_bins),
                "ece_platt": ece(y[:, c], p_cal[:, c], args.n_bins),
                "brier_raw": float(np.mean((pbar[:, c] - y[:, c]) ** 2)),
                "brier_platt": float(np.mean((p_cal[:, c] - y[:, c]) ** 2)),
            })
        cal = pd.DataFrame(cal_rows)
        say(cal.round(4).to_string(index=False))
        say(f"macro ECE raw {cal['ece_raw'].mean():.4f} -> Platt {cal['ece_platt'].mean():.4f}")
        say("note: the Platt maps are fitted on out-of-fold probabilities of other folds, whose "
            "models saw the evaluated fold; treat the recalibrated numbers as slightly optimistic.")
        table.merge(cal, on="class").to_csv(Path(args.out_dir) / f"eval_{backbone}.csv", index=False)

        # 3. selective prediction
        say("\n-- 3. Selective prediction (per class, balanced accuracy on retained images) --")
        thresholds, margin, pred = [], np.zeros_like(pbar), np.zeros_like(y)
        for c in range(len(classes)):
            fpr, tpr, thr = roc_curve(y[:, c], pbar[:, c])
            t = float(np.clip(thr[1:][np.argmax((tpr - fpr)[1:])], EPS, 1 - EPS))
            thresholds.append(t)
            pred[:, c] = pbar[:, c] >= t
            margin[:, c] = np.abs(logit(pbar[:, c]) - logit(t))  # distance to the decision threshold
        say("decision thresholds (Youden's J on the out-of-fold probabilities): "
            + ", ".join(f"{n.replace('dz_', '')} {t:.3f}" for n, t in zip(classes, thresholds)))

        def bal_acc(yc, pc):
            pos, neg = yc == 1, yc == 0
            if pos.sum() < 5 or neg.sum() < 5:
                return np.nan
            return 0.5 * (pc[pos].mean() + (1 - pc[neg]).mean())

        sel_rows = []
        for cov in args.coverages:
            n_keep = int(round(cov * len(y)))
            informed, rand = [], []
            for c in range(len(classes)):
                keep = np.argsort(-margin[:, c], kind="stable")[:n_keep]  # most confident first
                v = bal_acc(y[keep, c], pred[keep, c])
                if np.isnan(v):
                    continue
                informed.append(v)
                rand.append(np.nanmean([bal_acc(y[k, c], pred[k, c]) for k in
                                        (rng.choice(len(y), n_keep, replace=False) for _ in range(20))]))
            sel_rows.append({"coverage": cov, "n_kept": n_keep, "classes": len(informed),
                             "balacc_uncertainty": float(np.mean(informed)),
                             "balacc_random": float(np.mean(rand))})
        say(pd.DataFrame(sel_rows).round(3).to_string(index=False))
        say("per class, the images closest to the decision threshold are set aside first. A gain "
            "above the random column is what selective prediction has to show. The thresholds are "
            "chosen on the same out-of-fold probabilities, which is optimistic but applies equally "
            "to the random column. Classes with <5 positives or negatives left are skipped.")

        # 4. per manufacturer
        say("\n-- 4. Per manufacturer (macro AUROC over classes with >=10 positives and negatives) --")
        mf = manifest.loc[in_cohort, "Manufacturer"].fillna("(blank)").astype(str).to_numpy()
        for name in pd.Series(mf).value_counts().index:
            mask = mf == name
            if mask.sum() < args.min_manufacturer:
                continue
            say(f"{name}: images {int(mask.sum())}, macro AUROC "
                f"{macro_auc(y[mask], pbar[mask], 10, 10):.3f}")

        # 5. excluded images
        if "outside_by_fold" in runs[0].files:
            say("\n-- 5. Excluded images: model uncertainty vs main cohort --")
            stack = runs[0]["outside_by_fold"]  # (folds, outside rows, classes), seed 0
            n_out = stack.shape[1]
            pick = np.arange(n_out) % stack.shape[0]  # one single-model prediction per image
            p_out = stack[pick, np.arange(n_out)]
            u_out = entropy(p_out).mean(axis=1)
            u_in = entropy(runs[0]["probs"][in_cohort]).mean(axis=1)
            say(f"mean uncertainty: main cohort {u_in.mean():.4f}, excluded {u_out.mean():.4f}")
            say(f"AUROC of uncertainty for 'excluded vs main cohort': "
                f"{roc_auc_score(np.r_[np.zeros(len(u_in)), np.ones(n_out)], np.r_[u_in, u_out]):.3f} "
                f"(0.5 = no signal)")
            out_rows = manifest[~in_cohort]
            for col in ("quality_exclude", "tag_exclude"):
                if col in out_rows.columns:
                    flag = as_bool(out_rows[col]).to_numpy()
                    if 0 < flag.sum() < n_out:
                        sub = np.r_[np.zeros(len(u_in)), np.ones(int(flag.sum()))]
                        say(f"  {col}=True ({int(flag.sum())} images): AUROC "
                            f"{roc_auc_score(sub, np.r_[u_in, u_out[flag]]):.3f}")
            say("both sides use a single model's prediction per image, so the comparison is fair. "
                "Excluded images differ in other ways too (quality, projection), so a positive "
                "result shows a link, not a cause.")

    Path(args.out_txt).write_text("\n".join(LINES) + "\n")
    say(f"\nwritten: {args.out_txt}, {args.out_dir}/eval_<backbone>.csv")


if __name__ == "__main__":
    main()