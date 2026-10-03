"""
This code summarizes the VetXRay annotation table:

Reads the metadata spreadsheet, adds normalized helper columns(species scope, cleaned quality label, tag list)
and writes a plain-text summary with row counts and consistency checks.

Usage:
    python scripts/01_eda.py --raw_dir data/raw --out results/eda_summary.txt
"""

import argparse
import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = {"FileName", "PatientName", "breed", "specie","Projection", "Quality", "TAG", "NOTE"}

IN_SCOPE_SPECIES = ["Dog", "Cat"]
EXPECTED_IN_SCOPE = 9882 # ROW COUNT STATED IN THE DATASET DESCRIPTION
XLSX_NAME = "File list with tags.xlsx"
lines = []
NON_DISEASE_TAGS = {"no_finding", "exclude"}
MIN_POSITIVES = 50
EXPECTED_DISEASES_TAGS = 17 # NUMBER OF LESION TAGS STATED IN THE DATASET DESCRIPTION

def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize the VetXRay metadata table"  
    )
    parser.add_argument('--raw_dir', metavar='PATH', type=Path, 
         help="folder containing the dataset (default: data/raw)", default=Path('data/raw'))
    parser.add_argument('--out', metavar='PATH', type=Path, 
                         help="path of the text summary to write", default=Path('results/eda_summary.txt'))
    parser.add_argument('--xlsx', metavar='PATH', type=Path, 
                        help="annotation spreadsheet (default: <raw_dir>/File list with tags.xlsx)", default=None)
    args = parser.parse_args()

    if args.xlsx is None:
        args.xlsx = args.raw_dir / XLSX_NAME

    return args

def say_table(obj):
    say(obj.to_string())

def has_tag(df, tag):
    return df["tag_list"].apply(lambda tags: tag in tags)

def say(msg):
    print(msg)
    lines.append(msg)

def section(title):
    say("")
    say(f"=== {title} ===")

def load_table(xlsx_path):
    table = pd.read_excel(xlsx_path, engine="openpyxl")
    missed_cols = REQUIRED_COLUMNS - set(table.columns)
    if len(missed_cols) > 0:
        raise ValueError(f"Missing columns in {xlsx_path}: {sorted(missed_cols)}")
    return table

def add_clean_columns(df):
    df = df.copy()
    df["in_scope"] = df["specie"].isin(IN_SCOPE_SPECIES)
    
    df['quality_clean'] = df['Quality'].str.strip().str.lower()
    df["tag_list"] = df["TAG"].fillna("").astype(str).str.split("|").apply(lambda parts: [p.strip() for p in parts if p.strip()])
     
    return df

def report_tags(df):
    d = df[df["in_scope"]]
    counts = d["tag_list"].explode().value_counts()
    disease_counts = counts.drop(labels=list(NON_DISEASE_TAGS), errors = "ignore")
    n_tags = d["tag_list"].apply(len)
    
    section("Tag counts")
    say_table(counts)
    
    section("Disease Tag")
    say(f"Found {len(disease_counts)} disease tags (expected {EXPECTED_DISEASES_TAGS})")
    if len(disease_counts) != EXPECTED_DISEASES_TAGS:
        say("Warning: count differs from dataset description")
    say(", ".join(sorted(disease_counts.index)))
    
    section("Tags per image")
    say_table(n_tags.value_counts().sort_index())    

    section("No finding overlap")
    nf = has_tag(d, "no_finding")
    say(f"no_finding rows: {int(nf.sum())}")
    say(f"no_finding with another tag: {int((nf & (n_tags > 1)).sum())}")
    say(f"no_finding with exclude: {int((nf & has_tag(d, 'exclude')).sum())}")

    section("Rare classes")
    say("kept (>= MIN_POSITIVES):")
    say_table(disease_counts[disease_counts >= MIN_POSITIVES])
    say("rare (< MIN_POSITIVES):")
    say_table(disease_counts[disease_counts < MIN_POSITIVES])

    section("Threshold sensitivity")
    for t in [50, 100, 150]:
        say(f"threshold {t}: {int((disease_counts >= t).sum())} classes kept")    

def report_basics(df):
    d = df[df["in_scope"]]

    section("Species(all rows)")
    say_table(df["specie"].value_counts(dropna=False))

    section("Species(in scope)")
    say_table(d["specie"].value_counts())

    section("Quality(raw vs clean)")
    say_table(df["Quality"].value_counts(dropna=False))
    say_table(df["quality_clean"].value_counts(dropna=False))

    section("Projection(in scope)")
    say_table(d["Projection"].value_counts(dropna=False))

    section("Quality * Projection")
    say_table(pd.crosstab(d["quality_clean"], d["Projection"], margins=True))

    section("Projection Percentage")
    say_table((pd.crosstab(d["quality_clean"], d["Projection"], normalize="columns") * 100).round(1))

    section("Sum of NaN")
    say_table(d[["Projection", "quality_clean"]].isna().sum())

def report_checks(df):
    d = df[df["in_scope"]]
    n_tags = d["tag_list"].apply(len)
    nf = has_tag(d, "no_finding")
    ex_tag = has_tag(d, "exclude")
    ex_q = d["quality_clean"] == "exclude"
    
    section("no_finding companions")
    companions = d.loc[nf & (n_tags > 1), "tag_list"].explode().value_counts()
    say_table(companions.drop("no_finding", errors="ignore"))
    
    section("exclude: quality vs tag")
    say_table(pd.crosstab(ex_q.rename("quality_exclude"), ex_tag.rename("tag_exclude")))
    
    section("repeated tags within a row")
    has_dup = d["tag_list"].apply(lambda tags: len(tags) != len(set(tags)))
    say(f"rows with a repeated tag: {int(has_dup.sum())}")
    say(f"  of which quality exclude: {int((has_dup & ex_q).sum())}")
    say(f"  of which tag exclude: {int((has_dup & ex_tag).sum())}")

    section("Projection vs bronchial/interstitial (% positive)")
    for tag in ["bronchial_pattern", "interstitial_pattern"]:
        say(tag)
        pct = pd.crosstab(d["Projection"], has_tag(d, tag).rename("positive"), normalize="index") * 100
        say_table(pct.round(1))
        
    section("NOTE overall (top 15)")
    say_table(d["NOTE"].value_counts(dropna=False).head(15))
    section("NOTE for exclude rows (top 15)")
    say_table(d.loc[ex_q | ex_tag, "NOTE"].value_counts(dropna=False).head(15))  
      
def report_patients(df):
    d = df[df["in_scope"]]

    section("PatientName (aggregate only)")
    say(f"missing PatientName: {int(d['PatientName'].isna().sum())}")
    per_patient = d.groupby("PatientName").size()
    say(f"unique PatientName values: {len(per_patient)}")
    say_table(per_patient.describe())
    say("names by number of images:")
    say_table(per_patient.value_counts().sort_index().head(10))
    say(f"largest groups (images per name): {per_patient.nlargest(5).tolist()}")
    mixed = d.groupby("PatientName")["specie"].nunique() > 1
    say(f"names used for both Dog and Cat: {int(mixed.sum())}")
    
def find_dicom_files(raw_dir):
    files = []
    for p in Path(raw_dir).glob("RX_?/**/*.dcm"):
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        files.append(p)
    return files

def report_disk(df, raw_dir):
    section("Table vs disk")
    files = find_dicom_files(raw_dir)
    if not files:
        say(f"WARNING: no .dcm files found under {raw_dir}")
        return

    names_disk = [p.name for p in files]
    set_disk = set(names_disk)
    d = df[df["in_scope"]]
    all_names = set(df["FileName"])
    scope_names = set(d["FileName"])

    say(f".dcm files on disk (excluding __MACOSX): {len(names_disk)}")
    say(f"unique file names on disk: {len(set_disk)}")
    say(f"duplicate names across folders: {len(names_disk) - len(set_disk)}")
    say(f"table rows: {len(df)}, unique FileName: {len(all_names)}")
    say(f"in-scope rows missing on disk: {len(scope_names - set_disk)}")
    say(f"all table rows missing on disk: {len(all_names - set_disk)}")
    say(f"disk files not in table: {len(set_disk - all_names)}")
    say(f"examples (disk only): {sorted(set_disk - all_names)[:5]}")
    say(f"examples (table only): {sorted(all_names - set_disk)[:5]}")
          
def main():
    args = parse_args()
    df = add_clean_columns(load_table(args.xlsx))
    say(f"rows: {len(df)}, columns: {len(df.columns)} \n in scope {IN_SCOPE_SPECIES}: {EXPECTED_IN_SCOPE} (expected: {EXPECTED_IN_SCOPE})")
    report_basics(df)
    report_tags(df)
    report_checks(df)
    report_patients(df)
    report_disk(df, args.raw_dir)
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")

    section("Smoke test")
    say_table(df["specie"].value_counts(dropna=False).head(3))
    say(f"cardiomegaly rows: {int(has_tag(df, 'cardiomegaly').sum())}")

if __name__ == "__main__":
    main()