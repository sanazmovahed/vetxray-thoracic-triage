"""Resize and cache the radiographs listed in the manifest.

Every image is read once from DICOM, contrast-stretched with a percentile
window, inverted if stored as MONOCHROME1, padded to a square (no distortion)
and resized. All images are stored in one uint8 array, so later steps never
touch the DICOM files again.

Outputs (in --out_dir):
    images_<size>.npy   array of shape (N, size, size), same row order as the manifest
    index_<size>.csv    FileName, ok, error (ok=False rows hold an all-zero image)

Usage:
    python scripts/04_cache_images.py --raw_dir data/raw --size 224
    python scripts/04_cache_images.py --limit 200 --workers 1
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from common import find_dicom_files

lines = []


def say(msg):
    print(msg)
    lines.append(msg)


def parse_args():
    parser = argparse.ArgumentParser(description="Resize and cache the radiographs.")
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/manifest.csv"),
                        help="manifest written by 03_build_manifest.py")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"),
                        help="folder containing the RX_1 ... RX_5 image folders")
    parser.add_argument("--out_dir", type=Path, default=Path("data/cache"),
                        help="where the cached array is written")
    parser.add_argument("--size", type=int, default=224,
                        help="side length of the square output image")
    parser.add_argument("--percentiles", type=float, nargs=2, default=[2.0, 98.0],
                        metavar=("LOW", "HIGH"),
                        help="percentile window for contrast stretching")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="number of processes (1 = no multiprocessing)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cache only the first N images (for quick tests)")
    parser.add_argument("--preview", type=Path, default=Path("results/cache_preview.png"),
                        help="contact sheet of random cached images")
    parser.add_argument("--out_txt", type=Path, default=Path("results/cache_summary.txt"),
                        help="aggregate text summary")
    return parser.parse_args()


def to_unit_range(pixels, low, high, invert):
    img = pixels.astype(np.float32)
    lo, hi = np.percentile(img, [low, high])
    if hi <= lo:
        raise ValueError("flat image")
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo)
    if invert:
        img = 1.0 - img
    return img


def pad_to_square(img):
    height, width = img.shape
    side = max(height, width)
    out = np.zeros((side, side), dtype=img.dtype)
    top = (side - height) // 2
    left = (side - width) // 2
    out[top:top + height, left:left + width] = img
    return out


def load_one(job):
    """Return (uint8 array or None, error name or None) for one image."""
    path, size, low, high = job
    try:
        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array
        if pixels.ndim != 2:
            raise ValueError("expected a single-channel 2-D image")
        invert = ds.get("PhotometricInterpretation") == "MONOCHROME1"
        img = pad_to_square(to_unit_range(pixels, low, high, invert))
        img8 = (img * 255).round().astype(np.uint8)
        resized = Image.fromarray(img8).resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(resized), None
    except Exception as exc:
        return None, type(exc).__name__


def run_jobs(jobs, workers):
    if workers <= 1:
        yield from map(load_one, jobs)
    else:
        with Pool(workers) as pool:
            yield from pool.imap(load_one, jobs, chunksize=16)


def save_preview(images, manifest, ok, path, n=16, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = np.flatnonzero(ok)
    if len(ids) == 0:
        return
    rng = np.random.default_rng(seed)
    ids = rng.choice(ids, size=min(n, len(ids)), replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), ids):
        row = manifest.iloc[i]
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{row['specie']} {row['Projection']} {row['quality']}", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    paths = {p.name: p for p in find_dicom_files(args.raw_dir)}
    missing = [name for name in manifest["FileName"] if name not in paths]
    if missing:
        raise SystemExit(f"{len(missing)} manifest files were not found under --raw_dir.")

    low, high = args.percentiles
    jobs = [(paths[name], args.size, low, high) for name in manifest["FileName"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(
        args.out_dir / f"images_{args.size}.npy", mode="w+", dtype=np.uint8,
        shape=(len(jobs), args.size, args.size),
    )
    ok = np.zeros(len(jobs), dtype=bool)
    errors = [None] * len(jobs)

    for i, (img, error) in enumerate(run_jobs(jobs, args.workers)):
        if img is None:
            errors[i] = error
        else:
            images[i] = img
            ok[i] = True
        if (i + 1) % 500 == 0:
            print(f"cached {i + 1}/{len(jobs)}")
    images.flush()

    index = pd.DataFrame({"FileName": manifest["FileName"], "ok": ok, "error": errors})
    index.to_csv(args.out_dir / f"index_{args.size}.csv", index=False)

    say(f"images in manifest: {len(jobs)}")
    say(f"cached successfully: {int(ok.sum())}")
    say(f"failed: {int((~ok).sum())}")
    if (~ok).any():
        say(f"failures by type: {index.loc[~ok, 'error'].value_counts().to_dict()}")
    say(f"array shape: {images.shape}, dtype: {images.dtype}")
    say(f"size: {args.size}x{args.size}, percentile window: {low}-{high}")
    sample = np.flatnonzero(ok)[:500]
    if len(sample):
        values = np.asarray(images[sample], dtype=np.float32)
        say(f"pixel mean/std over {len(sample)} images (0-255): "
            f"{values.mean():.1f} / {values.std():.1f}")

    save_preview(images, manifest, ok, args.preview)
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()