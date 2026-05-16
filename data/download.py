"""Download rMD17 dataset files from figshare into data/raw/rmd17/.

Usage:
    uv run python data/download.py --dataset rmd17 --molecule aspirin
    uv run python data/download.py --dataset rmd17 --molecule all
    uv run python data/download.py --dataset rmd17 --molecule aspirin --force

Per-molecule files are 67–175 MB. The official 5-fold CV splits (28 KB) are
fetched on first run and extracted into data/raw/rmd17/splits/.
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
RMD17_DIR = REPO_ROOT / "data" / "raw" / "rmd17"

# Hardcoded URL + expected byte size table for the rMD17 figshare article
# (DOI 10.6084/m9.figshare.12672038). Regenerate manually from the figshare
# API if URLs ever change.
RMD17_FILES: dict[str, tuple[str, int]] = {
    "aspirin": ("https://ndownloader.figshare.com/files/62265757", 153_601_803),
    "azobenzene": ("https://ndownloader.figshare.com/files/62265754", 175_180_782),
    "benzene": ("https://ndownloader.figshare.com/files/62265739", 88_801_794),
    "ethanol": ("https://ndownloader.figshare.com/files/62265733", 67_201_791),
    "malonaldehyde": ("https://ndownloader.figshare.com/files/62265736", 67_201_791),
    "naphthalene": ("https://ndownloader.figshare.com/files/62265751", 132_001_800),
    "paracetamol": ("https://ndownloader.figshare.com/files/62265760", 146_401_802),
    "salicylic": ("https://ndownloader.figshare.com/files/62265748", 117_601_798),
    "toluene": ("https://ndownloader.figshare.com/files/62265742", 110_401_797),
    "uracil": ("https://ndownloader.figshare.com/files/62265745", 88_801_794),
}
SPLITS_URL = "https://ndownloader.figshare.com/files/62265763"
SPLITS_SIZE = 28_478


def _stream_download(url: str, dest: Path, expected_size: int) -> None:
    """Stream bytes from `url` to `dest` with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tinymlip-downloader"})
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", expected_size))
        with (
            open(dest, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            while chunk := response.read(1 << 16):
                f.write(chunk)
                bar.update(len(chunk))

    actual = dest.stat().st_size
    if actual != expected_size:
        print(
            f"warning: {dest.name} is {actual} bytes, expected {expected_size}. "
            "If figshare changed the file, update RMD17_FILES."
        )


def download_molecule(name: str, *, force: bool = False) -> None:
    url, expected_size = RMD17_FILES[name]
    dest = RMD17_DIR / f"rmd17_{name}.npz"
    if dest.exists() and dest.stat().st_size == expected_size and not force:
        print(f"skip: {dest.name} already present ({expected_size // (1024 * 1024)} MB)")
        return
    _stream_download(url, dest, expected_size)


def ensure_splits(*, force: bool = False) -> None:
    splits_dir = RMD17_DIR / "splits"
    if splits_dir.exists() and any(splits_dir.iterdir()) and not force:
        return
    zip_path = RMD17_DIR / "splits.zip"
    _stream_download(SPLITS_URL, zip_path, SPLITS_SIZE)
    # The zip contains a top-level "splits/" directory, so extracting into
    # RMD17_DIR lands the CSVs at RMD17_DIR/splits/*.csv. extractall creates
    # the directory; no explicit mkdir needed.
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RMD17_DIR)
    zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download rMD17 molecule files and official CV splits from figshare."
    )
    parser.add_argument("--dataset", required=True, choices=["rmd17"])
    parser.add_argument(
        "--molecule",
        required=True,
        choices=[*RMD17_FILES.keys(), "all"],
        help="rMD17 molecule name, or 'all' for every molecule.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files are cached."
    )
    args = parser.parse_args()

    targets = list(RMD17_FILES.keys()) if args.molecule == "all" else [args.molecule]
    for name in targets:
        download_molecule(name, force=args.force)
    ensure_splits(force=args.force)

    total_mb = sum((RMD17_DIR / f"rmd17_{n}.npz").stat().st_size for n in targets) // (1024 * 1024)
    rel = RMD17_DIR.relative_to(REPO_ROOT)
    print(f"Downloaded {len(targets)} molecule(s) ({total_mb} MB) to {rel}. Splits present.")


if __name__ == "__main__":
    main()
