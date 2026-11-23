"""Download the public UCI extract. The CSV is not stored in git."""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from readmission.config import (
    DATA_RAW,
    DATA_URL,
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    ZIP_SHA256,
)

CSV_NAME = "diabetic_data.csv"
MAPPING_NAME = "IDs_mapping.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "readmission-portfolio/1.0 (reproducible research download)"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return
        except Exception as exc:  # network blips happen; retry then raise
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}") from last_error


def download_dataset(force: bool = False) -> Path:
    """Download and extract ``diabetic_data.csv``. Return its local path.

    The zip is checked against a pinned SHA-256 so a changed upstream file
    fails loudly instead of silently changing the analysis.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_RAW / CSV_NAME
    mapping_path = DATA_RAW / MAPPING_NAME
    if csv_path.exists() and mapping_path.exists() and not force:
        return csv_path

    zip_path = DATA_RAW / "dataset_diabetes.zip"
    _fetch(DATA_URL, zip_path)
    digest = _sha256(zip_path)
    if digest != ZIP_SHA256:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {DATA_URL}. Expected {ZIP_SHA256}, got {digest}."
        )

    extract_dir = DATA_RAW / "_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    extracted_csv = next(extract_dir.rglob(CSV_NAME))
    extracted_mapping = next(extract_dir.rglob(MAPPING_NAME))
    shutil.copy2(extracted_csv, csv_path)
    shutil.copy2(extracted_mapping, mapping_path)
    shutil.rmtree(extract_dir)
    return csv_path


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the encounter extract and check the published shape."""
    csv_path = path or download_dataset()
    frame = pd.read_csv(csv_path, low_memory=False)
    if frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            f"Unexpected shape {frame.shape}; expected {(EXPECTED_ROWS, EXPECTED_COLUMNS)}."
        )
    if not frame["encounter_id"].is_unique:
        raise ValueError("encounter_id is not unique.")
    return frame
