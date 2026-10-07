"""Build patient-grouped cross-validation folds for multi-label classification.

All images that share a value in the manifest column `group` always land in the
same fold, so no animal appears on both sides of a split. Folds are balanced
greedily: groups that carry rare lesions are placed first, each into the fold
whose class counts would end up closest to the target.

Images outside the main cohort (flagged `exclude`) get fold -1. They are never
used for training or validation and are kept for a separate analysis.

Outputs:
    data/interim/folds.csv         one row per manifest row, same order as the manifest
    data/interim/classes.json      lesion columns kept (>= --min_pos positives in the cohort)
    results/folds_summary.txt      counts per fold and leakage checks (aggregate numbers only)

Usage:
    python scripts/06_make_folds.py
    python scripts/06_make_folds.py --n_splits 5 --seeds 0 1 2 --min_pos 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def parse_args():
    p = argparse.ArgumentParser(description="Make grouped, label-balanced folds.")
    p.add_argument("--manifest", default="data/interim/manifest.csv")
    p.add_argument("--out_csv", default="data/interim/folds.csv")
    p.add_argument("--out_classes", default="data/interim/classes.json")
    p.add_argument("--out_txt", default="results/folds_summary.txt")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--min_pos", type=int, default=50,
                   help="keep lesion classes with at least this many positives in the cohort")
    return p.parse_args()


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def cohort_mask(manifest):
    if "main_cohort" in manifest.columns:
        return as_bool(manifest["main_cohort"])
    return ~(as_bool(manifest["quality_exclude"]) | as_bool(manifest["tag_exclude"]))


def assign_groups(group_ids, labels, n_splits, seed):
    """Greedy balanced assignment of groups to folds. Returns fold per row."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(group_ids, return_inverse=True)
    n_groups, n_classes = len(uniq), labels.shape[1]

    group_pos = np.zeros((n_groups, n_classes))
    np.add.at(group_pos, inverse, labels)
    group_size = np.bincount(inverse, minlength=n_groups).astype(float)

    total_pos = labels.sum(axis=0)
    target_pos = np.maximum(total_pos / n_splits, 1.0)
    target_size = group_size.sum() / n_splits

    rarity = (group_pos / np.maximum(total_pos, 1)).sum(axis=1)
    order = rng.permutation(n_groups)
    order = order[np.argsort(-rarity[order], kind="stable")]

    fold_pos = np.zeros((n_splits, n_classes))
    fold_size = np.zeros(n_splits)
    group_fold = np.zeros(n_groups, dtype=int)

    for g in order:
        # cost = how much the squared distance to the target grows if the group joins a fold
        before = (((fold_pos - target_pos) / target_pos) ** 2).sum(axis=1)
        after = (((fold_pos + group_pos[g] - target_pos) / target_pos) ** 2).sum(axis=1)
        cost = after - before
        cost += ((fold_size + group_size[g] - target_size) / target_size) ** 2 \
            - ((fold_size - target_size) / target_size) ** 2
        cost += rng.random(n_splits) * 1e-9
        best = int(np.argmin(cost))
        group_fold[g] = best
        fold_pos[best] += group_pos[g]
        fold_size[best] += group_size[g]

    return group_fold[inverse]


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    for col in ("group", "FileName"):
        if col not in manifest.columns:
            raise SystemExit(f"manifest has no '{col}' column; found: {list(manifest.columns)}")

    in_cohort = cohort_mask(manifest).to_numpy()
    cohort = manifest[in_cohort]

    disease_cols = [c for c in manifest.columns if c.startswith("dz_")]
    positives = cohort[disease_cols].sum().sort_values(ascending=False)
    kept = [c for c in positives.index if positives[c] >= args.min_pos]
    dropped = [c for c in positives.index if c not in kept]

    say("== Cohort ==")
    say(f"manifest rows: {len(manifest)}")
    say(f"main cohort: {int(in_cohort.sum())} images, {cohort['group'].nunique()} groups")
    say(f"outside cohort (fold -1): {int((~in_cohort).sum())}")
    say(f"classes kept (>= {args.min_pos} positives): {len(kept)}; dropped: {len(dropped)}")
    if not kept:
        raise SystemExit("no class reaches --min_pos; lower it or check the manifest")

    labels = cohort[kept].to_numpy(dtype=float)
    group_ids = cohort["group"].to_numpy()

    folds = pd.DataFrame({"row": np.arange(len(manifest)), "FileName": manifest["FileName"]})
    for seed in args.seeds:
        col = f"fold_s{seed}"
        folds[col] = -1
        folds.loc[in_cohort, col] = assign_groups(group_ids, labels, args.n_splits, seed)

        # leakage check: every group must sit in exactly one fold
        per_group = folds.loc[in_cohort].groupby(manifest.loc[in_cohort, "group"])[col].nunique()
        say(f"\n== Seed {seed} ==")
        say(f"groups split across more than one fold: {int((per_group > 1).sum())} (must be 0)")

        counts = pd.DataFrame({"images": folds.loc[in_cohort, col].value_counts().sort_index()})
        pos = cohort[kept].groupby(folds.loc[in_cohort, col].to_numpy()).sum().T
        pos.columns = [f"fold{c}" for c in pos.columns]
        say("images per fold:")
        say(counts.T.to_string())
        say("positives per class per fold:")
        say(pos.to_string())
        spread = (pos.max(axis=1) - pos.min(axis=1)) / pos.mean(axis=1)
        say(f"largest relative spread across folds (max-min)/mean: {spread.max():.2f}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.out_csv, index=False)
    Path(args.out_classes).write_text(json.dumps({"classes": kept, "dropped": dropped}, indent=2))
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).write_text("\n".join(LINES) + "\n")
    say(f"\nwritten: {args.out_csv}, {args.out_classes}, {args.out_txt}")


if __name__ == "__main__":
    main()