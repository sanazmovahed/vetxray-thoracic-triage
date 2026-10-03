"""Scan the DICOM headers of the VetXRay images.

Reads only the header of every .dcm file (no pixel data), stores selected
fields in a CSV file and prints aggregate statistics: identifier uniqueness,
image geometry, photometric interpretation and transfer syntax. A random
sample of images is also decoded to check that pixel data can be read.

Identifier values are never written to the text summary; only counts are.

Usage:
    python scripts/02_scan_dicom_headers.py --raw_dir data/raw
    python scripts/02_scan_dicom_headers.py --limit 200 --decode_sample 10
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import pydicom

from common import find_dicom_files

FIELDS = ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "Rows", "Columns",
          "PhotometricInterpretation", "BitsStored", "PixelRepresentation",
          "Manufacturer", "ManufacturerModelName", "ViewPosition"]
SUMMARY_COLUMNS = ["PhotometricInterpretation", "BitsStored", "PixelRepresentation",
                   "TransferSyntaxUID", "Manufacturer"]
ID_COLUMNS = ["PatientID", "StudyInstanceUID"]

lines = []


def say(msg):
    print(msg)
    lines.append(msg)


def section(title):
    say("")
    say(f"== {title} ==")


def say_table(obj):
    say(obj.to_string())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan DICOM headers and test pixel decoding."
    )
    parser.add_argument(
        "--raw_dir", type=Path, default=Path("data/raw"),
        help="folder containing the RX_1 ... RX_5 image folders (default: data/raw)",
    )
    parser.add_argument(
        "--out_csv", type=Path, default=Path("data/interim/dicom_meta.csv"),
        help="per-file header table (contains identifiers, never commit it)",
    )
    parser.add_argument(
        "--out_txt", type=Path, default=Path("results/dicom_scan_summary.txt"),
        help="aggregate text summary",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="scan only the first N files (for quick tests)",
    )
    parser.add_argument(
        "--decode_sample", type=int, default=30,
        help="number of images to decode as a readability test",
    )
    return parser.parse_args()


def read_header(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=FIELDS)
    row = {name: ds.get(name) for name in FIELDS}
    ts = ds.file_meta.get("TransferSyntaxUID")
    row["TransferSyntaxUID"] = str(ts) if ts is not None else None
    row["FileName"] = Path(path).name
    return row


def scan_headers(files, raw_dir):
    rows = []
    for i, path in enumerate(files, start=1):
        try:
            row = read_header(path)
            row["error"] = None
        except Exception as exc:
            row = {"FileName": path.name, "error": type(exc).__name__}
        row["folder"] = path.relative_to(raw_dir).parts[0]
        rows.append(row)
        if i % 1000 == 0:
            print(f"scanned {i}/{len(files)}")
    return pd.DataFrame(rows)


def report_summary(meta):
    ok = meta[meta["error"].isna()]

    section("Header scan")
    say(f"files scanned: {len(meta)}, unreadable: {len(meta) - len(ok)}")
    if len(ok) < len(meta):
        say_table(meta["error"].value_counts())
    say("files per folder:")
    say_table(meta["folder"].value_counts().sort_index())

    for col in ID_COLUMNS:
        section(f"{col} (aggregate only)")
        values = ok[col].fillna("").astype(str).str.strip()
        blank = values == ""
        counts = values[~blank].value_counts()
        say(f"missing or empty: {int(blank.sum())}")
        say(f"unique values: {len(counts)}")
        say_table(counts.describe())
        say("values by number of images:")
        say_table(counts.value_counts().sort_index().head(10))
        say(f"largest groups (images per value): {counts.head(5).tolist()}")

    section("Image size (rows, columns), top 5")
    sizes = ok.groupby(["Rows", "Columns"]).size().sort_values(ascending=False)
    say_table(sizes.head(5))

    for col in SUMMARY_COLUMNS:
        section(f"{col}, top 5")
        say_table(ok[col].value_counts(dropna=False).head(5))


def decode_test(files, raw_dir, n):
    section("Pixel decoding test")
    if n <= 0 or not files:
        say("skipped")
        return

    rng = random.Random(42)
    by_folder = {}
    for p in files:
        by_folder.setdefault(p.relative_to(raw_dir).parts[0], []).append(p)

    per_folder = max(1, n // len(by_folder))
    sample = []
    for folder in sorted(by_folder):
        group = sorted(by_folder[folder])
        sample += rng.sample(group, min(per_folder, len(group)))

    decoded = 0
    failures = {}
    dtypes = {}
    first_error = None
    for p in sample:
        try:
            array = pydicom.dcmread(p).pixel_array
            decoded += 1
            dtypes[str(array.dtype)] = dtypes.get(str(array.dtype), 0) + 1
        except Exception as exc:
            name = type(exc).__name__
            failures[name] = failures.get(name, 0) + 1
            if first_error is None:
                first_error = exc

    say(f"sampled {len(sample)} files ({per_folder} per folder), decoded: {decoded}")
    say(f"decoded dtypes: {dtypes}")
    if failures:
        say(f"failures by type: {failures}")
        print(f"first error message (console only): {first_error}")


def main():
    args = parse_args()
    files = sorted(find_dicom_files(args.raw_dir))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise SystemExit("No .dcm files found. Check --raw_dir.")
    say(f"found {len(files)} .dcm files")

    meta = scan_headers(files, args.raw_dir)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(args.out_csv, index=False)
    print(f"header table written to {args.out_csv}")

    report_summary(meta)
    decode_test(files, args.raw_dir, args.decode_sample)

    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()