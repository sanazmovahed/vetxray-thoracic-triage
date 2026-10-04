"""Build the image manifest used by all later steps.

Joins the annotation table with the DICOM header scan (02_scan_dicom_headers.py),
keeps the in-scope images (Dog and Cat) whose header could be read, adds one
binary column per lesion tag and assigns every image to a patient group.

Several candidate grouping keys are compared first, because a group key that
mixes different animals (or splits one animal) makes any patient-level split
unreliable. The manifest stores integer group codes only, no identifiers.

Usage:
    python scripts/03_build_manifest.py --raw_dir data/raw --group_by patient_id
"""

import argparse
from pathlib import Path

import pandas as pd

from common import XLSX_NAME, NON_DISEASE_TAGS, add_clean_columns, has_tag, load_table

GROUP_CHOICES = ["patient_id", "study_uid", "patient_name", "name_key"]

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
        description="Join table and DICOM scan, assign patient groups."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="folder containing the dataset (default: data/raw)",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="annotation spreadsheet (default: <raw_dir>/File list with tags.xlsx)",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/interim/dicom_meta.csv"),
        help="header table written by 02_scan_dicom_headers.py",
    )
    parser.add_argument(
        "--group_by",
        choices=GROUP_CHOICES,
        default="patient_id",
        help="key used to define patient groups",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("data/interim/manifest.csv"),
        help="manifest (integer group codes, no identifiers)",
    )
    parser.add_argument(
        "--out_txt",
        type=Path,
        default=Path("results/manifest_summary.txt"),
        help="aggregate text summary",
    )
    args = parser.parse_args()
    if args.xlsx is None:
        args.xlsx = args.raw_dir / XLSX_NAME
    return args


def merge_table_and_meta(scope, meta):
    try:
        return scope.merge(
            meta, on="FileName", how="left", validate="one_to_one", indicator=True
        )
    except pd.errors.MergeError:
        raise SystemExit(
            "FileName is not unique in the table (in-scope rows) or in the DICOM "
            "scan. Files with the same name exist in several folders, so the "
            "join needs the folder as well. Check 01_eda.py and 02 output."
        )


def clean_text(series):
    return series.fillna("").astype(str).str.strip()


def build_keys(df):
    name = clean_text(df["PatientName"])
    name_key = name + "|" + clean_text(df["specie"]) + "|" + clean_text(df["breed"])
    return {
        "patient_id": clean_text(df["PatientID"]),
        "study_uid": clean_text(df["StudyInstanceUID"]),
        "patient_name": name,
        "name_key": name_key.where(name != "", ""),
    }


def diagnose_key(df, label, values):
    present = values != ""
    row = {
        "key": label,
        "missing": int((~present).sum()),
        "groups": 0,
        "mean_images": 0.0,
        "max_images": 0,
        "mixed_species": 0,
        "multi_study": 0,
    }
    if present.any():
        sub = df[present].assign(_key=values[present])
        grouped = sub.groupby("_key")
        sizes = grouped.size()
        row.update(
            {
                "groups": len(sizes),
                "mean_images": round(float(sizes.mean()), 2),
                "max_images": int(sizes.max()),
                "mixed_species": int((grouped["specie"].nunique() > 1).sum()),
                "multi_study": int((grouped["StudyInstanceUID"].nunique() > 1).sum()),
            }
        )
    return row


def assign_groups(df, keys, group_by):
    """Integer group codes; a missing key falls back to the study, then to the file."""
    primary = keys[group_by]
    study = keys["study_uid"]

    label = ("k:" + primary).where(primary != "", "")
    use_study = (label == "") & (study != "")
    label = label.where(~use_study, "s:" + study)
    use_file = label == ""
    label = label.where(~use_file, "f:" + df["FileName"].astype(str))

    codes, _ = pd.factorize(label)
    return codes, int(use_study.sum()), int(use_file.sum())


def build_manifest(usable, diseases, groups):
    manifest = pd.DataFrame(
        {
            "FileName": usable["FileName"],
            "folder": usable["folder"],
            "specie": usable["specie"],
            "Projection": usable["Projection"],
            "quality": usable["quality_clean"],
            "quality_exclude": usable["quality_clean"] == "exclude",
            "tag_exclude": has_tag(usable, "exclude"),
            "no_finding": has_tag(usable, "no_finding"),
            "Rows": pd.to_numeric(usable["Rows"], errors="coerce").astype("Int64"),
            "Columns": pd.to_numeric(usable["Columns"], errors="coerce").astype(
                "Int64"
            ),
            "PhotometricInterpretation": usable["PhotometricInterpretation"],
            "group": groups,
        }
    )
    disease_cols = []
    for tag in diseases:
        col = f"dz_{tag}"
        manifest[col] = has_tag(usable, tag).astype(int)
        disease_cols.append(col)
    manifest["n_disease"] = manifest[disease_cols].sum(axis=1)
    return manifest, disease_cols


def main():
    args = parse_args()

    table = add_clean_columns(load_table(args.xlsx))
    scope = table[table["in_scope"]].reset_index(drop=True)
    meta = pd.read_csv(args.meta, dtype=str)
    merged = merge_table_and_meta(scope, meta)

    section("Join")
    say(f"in-scope table rows: {len(scope)}")
    no_file = merged["_merge"] == "left_only"
    unreadable = (merged["_merge"] == "both") & merged["error"].notna()
    usable_mask = (merged["_merge"] == "both") & merged["error"].isna()
    say(f"no DICOM header found for the row: {int(no_file.sum())}")
    say(f"header unreadable: {int(unreadable.sum())}")
    say(f"usable images: {int(usable_mask.sum())}")

    usable = merged[usable_mask].reset_index(drop=True)
    diseases = sorted(
        {t for tags in scope["tag_list"] for t in tags} - NON_DISEASE_TAGS
    )

    section("Projection (table) vs ViewPosition (DICOM)")
    say_table(
        pd.crosstab(usable["Projection"], usable["ViewPosition"].fillna("(blank)"))
    )

    section("Candidate group keys")
    keys = build_keys(usable)
    diag = pd.DataFrame(
        [diagnose_key(usable, label, values) for label, values in keys.items()]
    )
    say_table(diag.set_index("key"))
    say("mixed_species: groups containing both Dog and Cat images")
    say("multi_study: groups spanning more than one StudyInstanceUID")

    section("Chosen grouping")
    groups, n_study, n_file = assign_groups(usable, keys, args.group_by)
    say(f"group key: {args.group_by}")
    say(f"images: {len(usable)}, groups: {len(set(groups))}")
    say(f"rows grouped by study (key missing): {n_study}")
    say(f"rows grouped by file (no key at all): {n_file}")

    manifest, disease_cols = build_manifest(usable, diseases, groups)

    section("Lesion tags in the manifest (positives)")
    say(f"lesion tags: {len(disease_cols)}")
    say_table(manifest[disease_cols].sum().sort_values(ascending=False))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_csv, index=False)
    print(f"manifest written to {args.out_csv}")

    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
