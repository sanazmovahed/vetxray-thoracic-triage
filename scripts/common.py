"""Helpers shared by the VetXRay scripts."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "FileName",
    "PatientName",
    "breed",
    "specie",
    "Projection",
    "Quality",
    "TAG",
    "NOTE",
}

IN_SCOPE_SPECIES = ["Dog", "Cat"]
EXPECTED_IN_SCOPE = 9882  # ROW COUNT STATED IN THE DATASET DESCRIPTION
XLSX_NAME = "File list with tags.xlsx"
NON_DISEASE_TAGS = {"no_finding", "exclude"}
MIN_POSITIVES = 50
EXPECTED_DISEASE_TAGS = 17  # NUMBER OF LESION TAGS STATED IN THE DATASET DESCRIPTION


def find_dicom_files(raw_dir):
    files = []
    for p in Path(raw_dir).glob("RX_?/**/*.dcm"):
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        files.append(p)
    return files


def load_table(xlsx_path):
    table = pd.read_excel(xlsx_path, engine="openpyxl")
    missed_cols = REQUIRED_COLUMNS - set(table.columns)
    if missed_cols:
        raise ValueError(f"Missing columns in {xlsx_path}: {sorted(missed_cols)}")
    return table


def add_clean_columns(df):
    df = df.copy()
    df["in_scope"] = df["specie"].isin(IN_SCOPE_SPECIES)
    df["quality_clean"] = df["Quality"].str.strip().str.lower()
    df["tag_list"] = (
        df["TAG"]
        .fillna("")
        .astype(str)
        .str.split("|")
        .apply(lambda parts: [p.strip() for p in parts if p.strip()])
    )
    return df


def has_tag(df, tag):
    return df["tag_list"].apply(lambda tags: tag in tags)
