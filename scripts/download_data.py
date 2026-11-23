"""Download the UCI Diabetes 130-US hospitals extract into data/raw/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from readmission.download import download_dataset, load_raw_dataset


def main() -> None:
    path = download_dataset()
    frame = load_raw_dataset(path)
    print(f"Saved {path} with {len(frame):,} encounters and {frame.shape[1]} columns.")


if __name__ == "__main__":
    main()
