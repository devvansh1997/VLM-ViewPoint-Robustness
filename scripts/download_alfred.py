"""
Download ALFRED validation JSON data to the datasets/ directory.

Downloads only the JSON metadata (no RGB frames) — these are the traj_data.json
files containing scene configs, start poses, NL instructions, and action plans.
We render our own frames via AI2-THOR so we don't need ALFRED's pre-recorded images.

Source: http://data.csail.mit.edu/janner/alfred/data/json_2.1.0.7z
Size:   ~2.7 GB compressed, ~4.5 GB extracted (all splits)
        We extract only valid_seen/ and valid_unseen/ (~300 MB total)

Usage (from repo root):
    python scripts/download_alfred.py
    python scripts/download_alfred.py --output_dir ../datasets
"""

import argparse
import os
import sys
import time
from pathlib import Path

ALFRED_JSON_URL = "https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z"
ARCHIVE_NAME    = "json_2.1.0.7z"

# Only extract these splits — skip train/ to save ~4GB
EXTRACT_SPLITS  = ["valid_seen", "valid_unseen"]


def download_file(url: str, dest_path: str) -> None:
    """Download a file with a progress bar."""
    try:
        import requests
    except ImportError:
        print("[download] 'requests' not installed. Run: pip install requests")
        sys.exit(1)

    print(f"[download] Downloading {url}")
    print(f"[download] Destination: {dest_path}")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    start = time.time()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    elapsed = time.time() - start
                    speed = downloaded / elapsed / (1024 * 1024)
                    print(
                        f"\r  {pct:5.1f}%  {downloaded/(1024**3):.2f}/{total/(1024**3):.2f} GB  "
                        f"{speed:.1f} MB/s",
                        end="",
                        flush=True,
                    )

    print()
    print(f"[download] Complete: {dest_path}")


def extract_validation_splits(archive_path: str, output_dir: str) -> str:
    """
    Extract only valid_seen/ and valid_unseen/ from the archive.
    Returns the path to the extracted data root (json_2.1.0/).
    """
    try:
        import py7zr
    except ImportError:
        print("[extract] 'py7zr' not installed. Run: pip install py7zr")
        sys.exit(1)

    print(f"[extract] Opening archive: {archive_path}")
    print(f"[extract] Extracting splits: {EXTRACT_SPLITS}")

    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        all_files = archive.getnames()

        # Filter to only valid_seen and valid_unseen
        to_extract = [
            f for f in all_files
            if any(f.startswith(f"json_2.1.0/{split}/") or
                   f == f"json_2.1.0/{split}"
                   for split in EXTRACT_SPLITS)
        ]

        if not to_extract:
            print("[extract] WARNING: No matching files found. Extracting everything.")
            archive.extractall(path=output_dir)
        else:
            print(f"[extract] Found {len(to_extract)} files to extract")
            archive.extract(path=output_dir, targets=to_extract)

    extracted_root = os.path.join(output_dir, "json_2.1.0")
    print(f"[extract] Done. Data at: {extracted_root}")
    return extracted_root


def count_episodes(data_root: str) -> None:
    """Print a summary of extracted episodes."""
    from pathlib import Path
    total = 0
    for split in EXTRACT_SPLITS:
        split_dir = Path(data_root) / split
        if split_dir.exists():
            n = len(list(split_dir.rglob("traj_data.json")))
            print(f"  {split}: {n} episodes")
            total += n
    print(f"  Total: {total} episodes")


def main():
    parser = argparse.ArgumentParser(description="Download ALFRED validation JSON data.")
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parents[2] / "datasets"),
        help="Directory to place the extracted data (default: ../datasets relative to repo)",
    )
    parser.add_argument(
        "--keep_archive",
        action="store_true",
        help="Keep the .7z archive after extraction (default: delete it)",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download if archive already exists",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME
    data_root    = output_dir / "json_2.1.0"

    # Check if already extracted
    if data_root.exists():
        splits_present = [s for s in EXTRACT_SPLITS if (data_root / s).exists()]
        if len(splits_present) == len(EXTRACT_SPLITS):
            print(f"[download] Data already extracted at {data_root}")
            print("[download] Episode summary:")
            count_episodes(str(data_root))
            return

    # Download
    if archive_path.exists() and args.skip_download:
        print(f"[download] Archive already exists at {archive_path} — skipping download.")
    else:
        download_file(ALFRED_JSON_URL, str(archive_path))

    # Extract
    extract_validation_splits(str(archive_path), str(output_dir))

    # Clean up archive unless asked to keep it
    if not args.keep_archive and archive_path.exists():
        os.remove(archive_path)
        print(f"[download] Removed archive: {archive_path}")

    # Summary
    print("\n[download] Episode summary:")
    count_episodes(str(data_root))
    print(f"\n[download] All done. Next step:")
    print(f"  python scripts/build_candidate_list.py --alfred_data {data_root}")


if __name__ == "__main__":
    main()
