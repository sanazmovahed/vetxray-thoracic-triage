"""Extract frozen embeddings from the cached radiographs.

Reads the uint8 array written by 04_cache_images.py, repeats the single channel
three times, normalizes with the statistics of the chosen pretrained model and
stores one feature vector per image. The backbone is never trained.

Two backbones are supported:
    imagenet   supervised ImageNet classifier (global-average-pooled features)
    dinov2     self-supervised DINOv2 vision transformer (class-token features)

Outputs (in --out_dir):
    features_<backbone>_<size>.npy    float32 array (N, D), same row order as the cache
    features_<backbone>_<size>.json   model name, library versions, normalization

Usage:
    python scripts/05_extract_embeddings.py --backbone imagenet --limit 64
    python scripts/05_extract_embeddings.py --backbone dinov2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch

BACKBONES = {
    "imagenet": {"model": "resnet50.a1_in1k", "kwargs": {}},
    "dinov2": {"model": "vit_base_patch14_dinov2.lvd142m", "kwargs": {"img_size": 224}},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frozen backbone embeddings.")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True,
                        help="which frozen backbone to use")
    parser.add_argument("--model_name", default=None,
                        help="override the timm model name of the chosen backbone")
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache"),
                        help="folder with images_<size>.npy and index_<size>.csv")
    parser.add_argument("--size", type=int, default=224, help="cached image size")
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/manifest.csv"),
                        help="manifest, used to verify the row order of the cache")
    parser.add_argument("--out_dir", type=Path, default=Path("data/features"),
                        help="where the features are written")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N images (for quick tests)")
    parser.add_argument("--amp", action="store_true",
                        help="use mixed precision on the GPU")
    return parser.parse_args()


def load_model(backbone, model_name, device):
    spec = BACKBONES[backbone]
    name = model_name or spec["model"]
    try:
        model = timm.create_model(name, pretrained=True, num_classes=0, **spec["kwargs"])
    except Exception as exc:
        candidates = timm.list_models(name.split(".")[0] + "*", pretrained=True)[:15]
        raise SystemExit(
            f"Could not create '{name}' ({type(exc).__name__}: {exc}).\n"
            f"Available names starting with '{name.split('.')[0]}': {candidates}\n"
            f"Pass one of them with --model_name."
        )
    model.eval().to(device)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    return model, mean, std, name, cfg


@torch.inference_mode()
def extract(model, images, mean, std, batch_size, device, amp):
    features = []
    started = time.time()
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(np.asarray(images[start:start + batch_size])).to(device)
        batch = batch.float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
        batch = (batch - mean) / std
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(batch)
        features.append(output.float().cpu().numpy())
        done = start + len(batch)
        if (start // batch_size) % 20 == 0:
            print(f"{done}/{len(images)} images, {time.time() - started:.0f}s")
    return np.concatenate(features)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found, extraction will be slow.")

    images = np.load(args.cache_dir / f"images_{args.size}.npy", mmap_mode="r")
    index = pd.read_csv(args.cache_dir / f"index_{args.size}.csv")
    if len(index) != len(images):
        raise SystemExit("images and index have different lengths.")
    if args.manifest.exists():
        manifest_names = pd.read_csv(args.manifest, usecols=["FileName"])["FileName"]
        if not manifest_names.head(len(index)).equals(index["FileName"].head(len(manifest_names))):
            raise SystemExit("cache row order does not match the manifest.")
    if args.limit is not None:
        images = images[:args.limit]

    model, mean, std, name, cfg = load_model(args.backbone, args.model_name, device)
    print(f"model: {name}, device: {device}, images: {len(images)}")

    features = extract(model, images, mean, std, args.batch_size, device, args.amp)
    finite = bool(np.isfinite(features).all())
    print(f"features: {features.shape}, all finite: {finite}")
    if not finite:
        raise SystemExit("non-finite values in the features.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"features_{args.backbone}_{args.size}"
    np.save(args.out_dir / f"{stem}.npy", features.astype(np.float32))
    meta = {
        "backbone": args.backbone, "model_name": name, "images": int(len(features)),
        "feature_dim": int(features.shape[1]), "mean": list(cfg["mean"]),
        "std": list(cfg["std"]), "amp": bool(args.amp),
        "timm": timm.__version__, "torch": torch.__version__,
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"written: {args.out_dir / stem}.npy")


if __name__ == "__main__":
    main()