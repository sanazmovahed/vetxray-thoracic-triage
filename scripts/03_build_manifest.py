"""Build the image manifest used by all later steps.

Joins the annotation table with the DICOM header scan (02_scan_dicom_headers.py),
keeps the in-scope images (Dog and Cat) whose header could be read, adds one
binary column per lesion tag and assigns every image to a patient group.

Several candidate grouping keys are compared first (including "union", which links
every pair of images that share a PatientID, a name+species+breed combination or a
StudyInstanceUID, so that an animal whose identifiers differ between visits still ends up
in a single group), because a group key that
mixes different animals (or splits one animal) makes any patient-level split
unreliable. Label prevalence is also reported per scanner and per image folder,
to expose acquisition-related confounding. The manifest stores integer group
codes only, no identifiers.

Usage:
    python scripts/03_build_manifest.py --raw_dir data/raw --group_by patient_id
"""

import argparse
from pathlib import Path

import pandas as pd

from common import (MIN_POSITIVES, NON_DISEASE_TAGS, REQUIRED_COLUMNS, XLSX_NAME,
                    add_clean_columns, has_tag, load_table)

GROUP_CHOICES = ["patient_id", "study_uid", "patient_name", "name_key", "union"]
UNION_KEYS = ("patient_id", "name_key", "study_uid")

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
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"),
                        help="folder containing the dataset (default: data/raw)")
    parser.add_argument("--xlsx", type=Path, default=None,
                        help="annotation spreadsheet (default: <raw_dir>/File list with tags.xlsx)")
    parser.add_argument("--meta", type=Path, default=Path("data/interim/dicom_meta.csv"),
                        help="header table written by 02_scan_dicom_headers.py")
    parser.add_argument("--group_by", choices=GROUP_CHOICES, default="patient_id",
                        help="key used to define patient groups")
    parser.add_argument("--out_csv", type=Path, default=Path("data/interim/manifest.csv"),
                        help="manifest (integer group codes, no identifiers)")
    parser.add_argument("--out_txt", type=Path, default=Path("results/manifest_summary.txt"),
                        help="aggregate text summary")
    args = parser.parse_args()
    if args.xlsx is None:
        args.xlsx = args.raw_dir / XLSX_NAME
    return args


def resolve_table_duplicates(scope):
    """Drop exact duplicate rows; drop every row of a FileName whose duplicates disagree."""
    deduped = scope.drop_duplicates(subset=sorted(REQUIRED_COLUMNS))
    n_exact = len(scope) - len(deduped)
    conflict = deduped["FileName"].duplicated(keep=False)
    names = sorted(deduped.loc[conflict, "FileName"].unique())
    return deduped[~conflict].reset_index(drop=True), n_exact, int(conflict.sum()), names


def merge_table_and_meta(scope, meta):
    try:
        return scope.merge(meta, on="FileName", how="left",
                           validate="one_to_one", indicator=True)
    except pd.errors.MergeError:
        raise SystemExit(
            "FileName is not unique in the DICOM scan: files with the same name exist "
            "in several folders, so the join would need the folder as well."
        )


def clean_text(series):
    return series.fillna("").astype(str).str.strip()


def build_keys(df):
    name = clean_text(df["PatientName"])
    name_key = (name + "|" + clean_text(df["specie"]) + "|" + clean_text(df["breed"]))
    return {
        "patient_id": clean_text(df["PatientID"]),
        "study_uid": clean_text(df["StudyInstanceUID"]),
        "patient_name": name,
        "name_key": name_key.where(name != "", ""),
    }


def diagnose_key(df, label, values):
    present = values != ""
    row = {"key": label, "missing": int((~present).sum()), "groups": 0,
           "mean_images": 0.0, "max_images": 0, "mixed_species": 0,
           "multi_study": 0, "multi_name": 0, "multi_id": 0}
    if present.any():
        sub = df[present].assign(_key=values[present])
        grouped = sub.groupby("_key")
        sizes = grouped.size()
        row.update({
            "groups": len(sizes),
            "mean_images": round(float(sizes.mean()), 2),
            "max_images": int(sizes.max()),
            "mixed_species": int((grouped["specie"].nunique() > 1).sum()),
            "multi_study": int((grouped["StudyInstanceUID"].nunique() > 1).sum()),
            "multi_name": int((grouped["PatientName"].nunique() > 1).sum()),
            "multi_id": int((grouped["PatientID"].nunique() > 1).sum()),
        })
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


def union_groups(df, keys, names=UNION_KEYS):
    """Integer codes of the connected components of images linked by any shared key."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for name in names:
        first = {}
        for i, value in enumerate(keys[name].tolist()):
            if value == "":
                continue
            if value in first:
                root_a, root_b = find(first[value]), find(i)
                if root_a != root_b:
                    parent[root_b] = root_a
            else:
                first[value] = i

    codes, _ = pd.factorize(pd.Series([find(i) for i in range(len(df))]))
    return codes


def build_manifest(usable, diseases, groups):
    manifest = pd.DataFrame({
        "FileName": usable["FileName"],
        "folder": usable["folder"],
        "specie": usable["specie"],
        "Projection": usable["Projection"],
        "quality": usable["quality_clean"],
        "quality_exclude": usable["quality_clean"] == "exclude",
        "tag_exclude": has_tag(usable, "exclude"),
        "no_finding": has_tag(usable, "no_finding"),
        "Manufacturer": usable["Manufacturer"],
        "BitsStored": pd.to_numeric(usable["BitsStored"], errors="coerce").astype("Int64"),
        "Rows": pd.to_numeric(usable["Rows"], errors="coerce").astype("Int64"),
        "Columns": pd.to_numeric(usable["Columns"], errors="coerce").astype("Int64"),
        "PhotometricInterpretation": usable["PhotometricInterpretation"],
        "group": groups,
    })
    manifest["main_cohort"] = ~(manifest["tag_exclude"] | manifest["quality_exclude"])
    disease_cols = []
    for tag in diseases:
        col = f"dz_{tag}"
        manifest[col] = has_tag(usable, tag).astype(int)
        disease_cols.append(col)
    manifest["n_disease"] = manifest[disease_cols].sum(axis=1)
    return manifest, disease_cols


def report_confounding(manifest, disease_cols):
    frequent = [c for c in disease_cols if manifest[c].sum() >= MIN_POSITIVES]
    flags = ["no_finding", "tag_exclude", "quality_exclude"]

    for col in ["Manufacturer", "folder"]:
        section(f"Images and label prevalence by {col}")
        say_table(manifest[col].value_counts(dropna=False))
        say("percent of images carrying each label:")
        rates = manifest.groupby(col, dropna=False)[frequent + flags].mean() * 100
        say_table(rates.round(1).T)

    section("Manufacturer vs folder")
    say_table(pd.crosstab(manifest["Manufacturer"], manifest["folder"]))
    section("BitsStored vs Manufacturer")
    say_table(pd.crosstab(manifest["BitsStored"], manifest["Manufacturer"]))

    section("exclude flags")
    say("quality (rows) vs tag exclude (columns):")
    say_table(pd.crosstab(manifest["quality"], manifest["tag_exclude"]))
    only_tag = manifest["tag_exclude"] & ~manifest["quality_exclude"]
    say(f"tag exclude without quality exclude: {int(only_tag.sum())}")
    with_lesion = manifest["tag_exclude"] & (manifest["n_disease"] > 0)
    say(f"tag exclude rows that also carry a lesion tag: {int(with_lesion.sum())}")
    for col in ["Projection", "specie"]:
        say(f"percent tag exclude by {col}:")
        say_table((manifest.groupby(col)["tag_exclude"].mean() * 100).round(1))


def report_cohort(manifest, disease_cols):
    cohort = manifest[manifest["main_cohort"]]
    section("Main cohort (no exclude flag)")
    say(f"images: {len(cohort)} of {len(manifest)}")
    say(f"patient groups: {cohort['group'].nunique()}")
    say("positives per lesion tag in the cohort:")
    say_table(cohort[disease_cols].sum().sort_values(ascending=False))


def report_group_sizes(groups):
    section("Group size distribution (chosen grouping)")
    sizes = pd.Series(groups).value_counts()
    say(f"groups: {len(sizes)}, mean images: {sizes.mean():.2f}, max: {int(sizes.max())}")
    say(f"five largest groups (images): {sizes.head(5).tolist()}")
    say("groups by number of images:")
    say_table(sizes.value_counts().sort_index().head(10))


def main():
    args = parse_args()

    table = add_clean_columns(load_table(args.xlsx))
    scope = table[table["in_scope"]].reset_index(drop=True)

    section("Table duplicates (in-scope rows)")
    say(f"in-scope table rows: {len(scope)}")
    scope, n_exact, n_conflict, names = resolve_table_duplicates(scope)
    say(f"exact duplicate rows removed: {n_exact}")
    say(f"rows removed because duplicates disagree: {n_conflict}")
    if names:
        say(f"conflicting file names (up to 5): {names[:5]}")
    say(f"in-scope rows kept: {len(scope)}")

    meta = pd.read_csv(args.meta, dtype=str)
    merged = merge_table_and_meta(scope, meta)

    section("Join")
    no_file = merged["_merge"] == "left_only"
    unreadable = (merged["_merge"] == "both") & merged["error"].notna()
    usable_mask = (merged["_merge"] == "both") & merged["error"].isna()
    say(f"no DICOM header found for the row: {int(no_file.sum())}")
    say(f"header unreadable: {int(unreadable.sum())}")
    say(f"usable images: {int(usable_mask.sum())}")

    usable = merged[usable_mask].reset_index(drop=True)
    diseases = sorted({t for tags in scope["tag_list"] for t in tags} - NON_DISEASE_TAGS)

    section("Projection (table) vs ViewPosition (DICOM)")
    say_table(pd.crosstab(usable["Projection"], usable["ViewPosition"].fillna("(blank)")))

    section("Candidate group keys")
    keys = build_keys(usable)
    union_codes = union_groups(usable, keys)
    candidates = dict(keys)
    candidates["union"] = pd.Series(union_codes, index=usable.index).astype(str)
    diag = pd.DataFrame([diagnose_key(usable, label, values) for label, values in candidates.items()])
    say_table(diag.set_index("key"))
    say("mixed_species: groups containing both Dog and Cat images")
    say("multi_study: groups spanning more than one StudyInstanceUID")
    say("multi_name / multi_id: groups with more than one PatientName / PatientID")

    section("Chosen grouping")
    if args.group_by == "union":
        groups, n_study, n_file = union_codes, 0, 0
    else:
        groups, n_study, n_file = assign_groups(usable, keys, args.group_by)
    say(f"group key: {args.group_by}")
    say(f"images: {len(usable)}, groups: {len(set(groups))}")
    say(f"rows grouped by study (key missing): {n_study}")
    say(f"rows grouped by file (no key at all): {n_file}")

    report_group_sizes(groups)
    manifest, disease_cols = build_manifest(usable, diseases, groups)

    section("Lesion tags in the manifest (positives)")
    say(f"lesion tags: {len(disease_cols)}")
    say_table(manifest[disease_cols].sum().sort_values(ascending=False))

    report_cohort(manifest, disease_cols)
    report_confounding(manifest, disease_cols)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_csv, index=False)
    print(f"manifest written to {args.out_csv}")

    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()