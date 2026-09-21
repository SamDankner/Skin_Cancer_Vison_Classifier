"""Resumable official acquisition for the Zenodo MCSI release."""
from __future__ import annotations

from hashlib import md5
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


MCSI_RECORD_URL = "https://zenodo.org/records/8360076"
MCSI_DOWNLOAD_URL = "https://zenodo.org/records/8360076/files/MCSI.zip?download=1"
MCSI_MD5 = "e8e490856d0a7e4871ae9e2dc15e81fe"
MCSI_EXPECTED_IMAGES = 400


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe path in MCSI archive: {member.filename!r}")
        bundle.extractall(destination)


def _images(directory: Path) -> list[Path]:
    return [path for path in directory.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]


def download_mcsi_dataset(raw_root: str | Path, *, opener=urlopen, chunk_size: int = 4 * 1024 * 1024) -> dict:
    """Download MCSI from Zenodo, verify its published checksum, and extract it."""
    directory = Path(raw_root) / "MCSI"
    directory.mkdir(parents=True, exist_ok=True)
    images = _images(directory)
    if len(images) >= MCSI_EXPECTED_IMAGES:
        return {"dataset": "MCSI", "status": "ready", "valid_images": len(images), "downloaded_bytes": 0, "official_source_url": MCSI_RECORD_URL, "checksum": MCSI_MD5}
    archive = directory / "MCSI.zip"
    partial = directory / "MCSI.zip.part"
    if not archive.is_file():
        offset = partial.stat().st_size if partial.is_file() else 0
        request = Request(MCSI_DOWNLOAD_URL, headers={"User-Agent": "Skin-Cancer-Classifier data preparation", **({"Range": f"bytes={offset}-"} if offset else {})})
        with opener(request, timeout=120) as response:
            append = bool(offset and getattr(response, "status", response.getcode()) == 206)
            with partial.open("ab" if append else "wb") as target:
                while block := response.read(chunk_size):
                    target.write(block)
        partial.replace(archive)
    digest = md5(archive.read_bytes()).hexdigest()
    if digest != MCSI_MD5:
        raise ValueError(f"MCSI checksum mismatch: expected {MCSI_MD5}, got {digest}")
    _safe_extract(archive, directory)
    images = _images(directory)
    return {"dataset": "MCSI", "status": "ready" if len(images) >= MCSI_EXPECTED_IMAGES else "incomplete", "valid_images": len(images), "downloaded_bytes": archive.stat().st_size, "archive": str(archive), "official_source_url": MCSI_RECORD_URL, "checksum": MCSI_MD5}


__all__ = ["MCSI_DOWNLOAD_URL", "MCSI_EXPECTED_IMAGES", "MCSI_RECORD_URL", "download_mcsi_dataset"]
