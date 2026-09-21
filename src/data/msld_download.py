"""Optional authenticated Kaggle acquisition for MSLD v2.0."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


MSLD_SLUG = "joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20"
MSLD_URL = f"https://www.kaggle.com/datasets/{MSLD_SLUG}"


def download_msld_dataset(raw_root: str | Path) -> dict:
    """Use the user's normal Kaggle CLI credentials when available.

    No credentials are created, scraped, or bypassed.  The manifest importer
    independently excludes every augmented folder after acquisition.
    """
    directory = Path(raw_root) / "MSLD_v2"
    directory.mkdir(parents=True, exist_ok=True)
    existing = list(directory.rglob("*"))
    if any(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for path in existing):
        return {"dataset": "MSLD_v2", "status": "ready", "downloaded": False, "official_source_url": MSLD_URL}
    executable = shutil.which("kaggle")
    if executable is None:
        return {"dataset": "MSLD_v2", "status": "kaggle_authentication_required", "downloaded": False, "official_source_url": MSLD_URL, "instruction": "Install/configure the normal Kaggle CLI and kaggle.json, then rerun prepare_data.py --all-development."}
    completed = subprocess.run(
        [executable, "datasets", "download", "-d", MSLD_SLUG, "-p", str(directory), "--unzip"],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        return {"dataset": "MSLD_v2", "status": "kaggle_authentication_required", "downloaded": False, "official_source_url": MSLD_URL, "detail": completed.stderr[-1000:]}
    return {"dataset": "MSLD_v2", "status": "ready", "downloaded": True, "official_source_url": MSLD_URL}


__all__ = ["MSLD_SLUG", "MSLD_URL", "download_msld_dataset"]
