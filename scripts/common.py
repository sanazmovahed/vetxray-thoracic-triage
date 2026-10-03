from  pathlib import Path

def find_dicom_files(raw_dir):
    files = []
    for p in Path(raw_dir).glob("RX_?/**/*.dcm"):
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        files.append(p)
    return files