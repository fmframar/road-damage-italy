"""Download and unzip the Italian road damage dataset from Zenodo.
Dataset: Giordani et al., Zenodo DOI 10.5281/zenodo.17834373 (MIT license).
"""
import sys
import zipfile
from pathlib import Path

import requests

RECORD_ID = "17834373"
OUT_DIR = Path("data/raw")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ask Zenodo which files the record contains
    meta = requests.get(f"https://zenodo.org/api/records/{RECORD_ID}", timeout=30)
    meta.raise_for_status()
    files = meta.json()["files"]
    print("Files in record:", [f["key"] for f in files])

    # 2. Download each file (stream, so big files don't fill memory)
    for f in files:
        name = f["key"]
        url = f["links"]["self"]
        target = OUT_DIR / name
        if target.exists():
            print(f"{name} already downloaded, skipping")
            continue
        print(f"Downloading {name} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(target, "wb") as out:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    out.write(chunk)

        # 3. Unzip if it's an archive
        if zipfile.is_zipfile(target):
            print(f"Unzipping {name} ...")
            with zipfile.ZipFile(target) as z:
                z.extractall(OUT_DIR)

    print("Done. Data in", OUT_DIR.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Automatic download failed:", e)
        print("Download manually from https://doi.org/10.5281/zenodo.17834373 "
              f"and unzip into {OUT_DIR}")
        sys.exit(1)
