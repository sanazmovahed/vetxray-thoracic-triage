"""Make the paper figures and the paired comparison of the two backbones.

Figures (PNG, 300 dpi) in --fig_dir:
    fig1_auroc.png         per-class AUROC with 95% intervals, both backbones
    fig2_reliability.png   reliability diagrams for a few classes, both backbones
    fig3_selective.png     balanced accuracy against coverage, both backbones

Text (aggregate numbers only):
    results/paired_summary.txt   macro AUROC difference (second backbone minus first)
                                 with a cluster-bootstrap interval, per class as well

Colours are fixed per backbone (never swapped between figures): blue = DINOv2,
orange = ResNet-50. Line style and marker shape differ too, so the figures do not
depend on colour alone.

Usage:
    python scripts/09_make_figures.py
    python scripts/09_make_figures.py --preds_dir data/preds_C0.01 --eval_dir results/C0.01 --fig_dir results/figures_C0.01
    # each backbone at its own C (reference = ResNet-50 at C=0.001, DINOv2 at C=0.01):
    python scripts/09_make_figures.py --preds_dir_ref data/preds_C0.001 --eval_dir_ref results/C0.001 \\
        --preds_dir data/preds_C0.01 --eval_dir results/C0.01 --fig_dir results/figures_bestC \\
        --out_txt results/paired_summary_bestC.txt
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-6
STYLE = {  # fixed per backbone
    "imagenet": {"label": "ResNet-50 (ImageNet)", "color": "#eb6834", "marker": "s", "ls": "--"},
    "dinov2": {"label": "DINOv2 ViT-B/14", "color": "#2a78d6", "marker": "o", "ls": "-"},
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser(description="Make paper figures and the paired backbone comparison.")
    p.add_argument("--manifest", default="data/interim/manifest.csv")
    p.add_argument("--classes", default="data/interim/classes.json")
    p.add_argument("--preds_dir", default="data/preds")
    p.add_argument("--preds_dir_ref", default=None,
                   help="predictions folder of the reference (first) backbone, if it was run with a "
                        "different C than the second one; default: same as --preds_dir")
    p.add_argument("--eval_dir_ref", default=None,
                   help="eval folder of the reference backbone; default: same as --eval_dir")
    p.add_argument("--eval_dir", default="results", help="folder with eval_<backbone>.csv from 08")
    p.add_argument("--fig_dir", default="results/figures")
    p.add_argument("--out_txt", default="results/paired_summary.txt")
    p.add_argument("--backbones", nargs=2, default=["imagenet", "dinov2"],
                   help="reference backbone first; the difference is second minus first")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--reliability_classes", nargs="+",
                   default=["dz_cardiomegaly", "dz_pleural_effusion", "dz_mass"])
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--rng_seed", type=int, default=0)
    return p.parse_args()


def setup_axes(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_probs(preds_dir, backbone, seeds, in_cohort_ref=None):
    runs = [np.load(Path(preds_dir) / f"oof_{backbone}_s{s}.npz") for s in seeds]
    fold0 = runs[0]["fold"]
    mask = fold0 >= 0
    return np.mean([r["probs"][mask] for r in runs], axis=0), mask


def fig_auroc(tables, backbones, fig_dir):
    order = tables[backbones[1]].sort_values("auroc").reset_index(drop=True)["class"].tolist()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    setup_axes(ax)
    ax.grid(axis="y", visible=False)
    for i, bb in enumerate(backbones):
        t = tables[bb].set_index("class").loc[order]
        y = np.arange(len(order)) + (-0.17 if i == 0 else 0.17)
        st = STYLE[bb]
        ax.errorbar(t["auroc"], y, xerr=[t["auroc"] - t["auroc_lo"], t["auroc_hi"] - t["auroc"]],
                    fmt=st["marker"], color=st["color"], ecolor=st["color"], elinewidth=1.4,
                    capsize=0, markersize=5.5, markeredgecolor="white", markeredgewidth=0.8,
                    label=st["label"])
    ax.axvline(0.5, color=INK2, linewidth=0.9, linestyle=":")
    ax.set_yticks(np.arange(len(order)))
    names = tables[backbones[1]].set_index("class").loc[order]
    ax.set_yticklabels([f"{c.replace('dz_', '').replace('_', ' ')} (n={int(names.loc[c, 'positives'])})"
                        for c in order], color=INK)
    ax.set_xlabel("AUROC (out-of-fold, 95% cluster-bootstrap interval)", color=INK2, fontsize=9)
    ax.set_xlim(0.45, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "fig1_auroc.png", dpi=300, facecolor="white")
    plt.close(fig)


def reliability_points(y, p, n_bins=10):
    order = np.argsort(p, kind="stable")
    xs, ys = [], []
    for part in np.array_split(order, n_bins):  # equal-count bins, stable for rare classes
        xs.append(p[part].mean())
        ys.append(y[part].mean())
    return np.array(xs), np.array(ys)


def fig_reliability(probs, labels, classes, backbones, chosen, fig_dir):
    chosen = [c for c in chosen if c in classes]
    fig, axes = plt.subplots(1, len(chosen), figsize=(3.4 * len(chosen), 3.4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, chosen):
        setup_axes(ax)
        c = classes.index(name)
        top = 0.0
        for bb in backbones:
            st = STYLE[bb]
            x, yv = reliability_points(labels[:, c], probs[bb][:, c])
            ax.plot(x, yv, st["ls"], color=st["color"], marker=st["marker"], linewidth=1.6,
                    markersize=5, markeredgecolor="white", markeredgewidth=0.7, label=st["label"])
            top = max(top, x.max(), yv.max())
        top = min(1.0, top * 1.1)
        ax.plot([0, top], [0, top], color=INK2, linewidth=0.9, linestyle=":")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_title(name.replace("dz_", "").replace("_", " "), color=INK, fontsize=10)
        ax.set_xlabel("mean predicted probability", color=INK2, fontsize=9)
    axes[0].set_ylabel("observed fraction positive", color=INK2, fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "fig2_reliability.png", dpi=300, facecolor="white")
    plt.close(fig)


def balanced_accuracy(y, pred):
    pos, neg = y == 1, y == 0
    if pos.sum() < 5 or neg.sum() < 5:
        return np.nan
    return 0.5 * (pred[pos].mean() + (1 - pred[neg]).mean())


def selective_curve(probs, labels, coverages, rng):
    n, n_classes = probs.shape
    margin = np.zeros_like(probs)
    pred = np.zeros(probs.shape, dtype=bool)
    for c in range(n_classes):
        fpr, tpr, thr = roc_curve(labels[:, c], probs[:, c])
        t = float(np.clip(thr[1:][np.argmax((tpr - fpr)[1:])], EPS, 1 - EPS))
        pred[:, c] = probs[:, c] >= t
        margin[:, c] = np.abs(logit(probs[:, c]) - logit(t))
    informed, random_ = [], []
    for cov in coverages:
        k = int(round(cov * n))
        inf, rnd = [], []
        for c in range(n_classes):
            keep = np.argsort(-margin[:, c], kind="stable")[:k]
            v = balanced_accuracy(labels[keep, c], pred[keep, c])
            if np.isnan(v):
                continue
            inf.append(v)
            rnd.append(np.nanmean([balanced_accuracy(labels[r, c], pred[r, c]) for r in
                                   (rng.choice(n, k, replace=False) for _ in range(10))]))
        informed.append(np.mean(inf))
        random_.append(np.mean(rnd))
    return np.array(informed), np.array(random_)


def fig_selective(probs, labels, backbones, fig_dir, rng):
    coverages = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    setup_axes(ax)
    for bb in backbones:
        st = STYLE[bb]
        inf, rnd = selective_curve(probs[bb], labels, coverages, rng)
        ax.plot(coverages, inf, st["ls"], color=st["color"], marker=st["marker"], linewidth=1.8,
                markersize=5, markeredgecolor="white", markeredgewidth=0.7, label=st["label"])
        ax.plot(coverages, rnd, color=st["color"], linewidth=1.0, linestyle=":", alpha=0.7)
    ax.plot([], [], color=INK2, linewidth=1.0, linestyle=":", label="set aside at random")
    ax.invert_xaxis()
    ax.set_xlabel("coverage (fraction of images the model decides)", color=INK2, fontsize=9)
    ax.set_ylabel("balanced accuracy (macro over classes)", color=INK2, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "fig3_selective.png", dpi=300, facecolor="white")
    plt.close(fig)


def paired_difference(probs, labels, groups, classes, backbones, n_boot, rng, lines):
    ref, new = backbones
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    n_classes = len(classes)
    diffs = np.full((n_boot, n_classes), np.nan)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        for c in range(n_classes):
            y = labels[idx, c]
            if 0 < y.sum() < len(y):
                diffs[b, c] = (roc_auc_score(y, probs[new][idx, c])
                               - roc_auc_score(y, probs[ref][idx, c]))
    point = np.array([roc_auc_score(labels[:, c], probs[new][:, c])
                      - roc_auc_score(labels[:, c], probs[ref][:, c]) for c in range(n_classes)])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5], axis=0)
    macro = np.nanmean(diffs, axis=1)
    mlo, mhi = np.nanpercentile(macro, [2.5, 97.5])
    lines.append(f"AUROC difference, {STYLE[new]['label']} minus {STYLE[ref]['label']} "
                 f"(same images, same groups resampled together, {n_boot} bootstrap draws)")
    lines.append(f"macro AUROC difference: {point.mean():+.3f}  (95% CI {mlo:+.3f} to {mhi:+.3f})")
    lines.append("per class:")
    for c, name in enumerate(classes):
        flag = "interval excludes 0" if (lo[c] > 0 or hi[c] < 0) else "interval includes 0"
        lines.append(f"  {name:28s} {point[c]:+.3f}  ({lo[c]:+.3f} to {hi[c]:+.3f})  {flag}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)
    manifest = pd.read_csv(args.manifest)
    classes = json.loads(Path(args.classes).read_text())["classes"]
    labels_all = manifest[classes].to_numpy(dtype=np.int8)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    probs, masks = {}, []
    for i, bb in enumerate(args.backbones):
        folder = args.preds_dir_ref if (i == 0 and args.preds_dir_ref) else args.preds_dir
        probs[bb], mask = load_probs(folder, bb, args.seeds)
        masks.append(mask)
    if not np.array_equal(masks[0], masks[1]):
        raise SystemExit("the two backbones use different cohorts; rerun 07 for both")
    labels = labels_all[masks[0]]
    groups = manifest["group"].to_numpy()[masks[0]]

    tables = {}
    for i, bb in enumerate(args.backbones):
        folder = args.eval_dir_ref if (i == 0 and args.eval_dir_ref) else args.eval_dir
        tables[bb] = pd.read_csv(Path(folder) / f"eval_{bb}.csv")
    fig_auroc(tables, args.backbones, args.fig_dir)
    fig_reliability(probs, labels, classes, args.backbones, args.reliability_classes, args.fig_dir)
    fig_selective(probs, labels, args.backbones, args.fig_dir, rng)

    lines = []
    paired_difference(probs, labels, groups, classes, args.backbones, args.n_boot, rng, lines)
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nfigures: {args.fig_dir}/fig1_auroc.png, fig2_reliability.png, fig3_selective.png")


if __name__ == "__main__":
    main()