"""Resumable acquisition of PAD-UFES-20 from its official Mendeley record."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


PAD_DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/zr7vgbcyr2/download/1"
PAD_EXPECTED_IMAGES = 2298


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe path in PAD-UFES-20 archive: {member.filename!r}")
        bundle.extractall(destination)


def _extract_nested_archives(directory: Path) -> None:
    """Extract the official three image-part ZIPs without trusting paths."""
    for archive in sorted(directory.rglob("*.zip")):
        if archive.name == "pad_ufes_20.zip":
            continue
        _safe_extract(archive, archive.parent)


def _pad_inventory(directory: Path) -> tuple[list[Path], list[Path]]:
    images = sorted(directory.rglob("*.png")) if directory.exists() else []
    metadata = sorted(directory.rglob("*.csv")) if directory.exists() else []
    return images, metadata


def download_pad_dataset(
    raw_root: str | Path,
    *,
    opener=urlopen,
    progress=None,
    chunk_size: int = 4 * 1024 * 1024,
) -> dict:
    """Download and safely extract the official CC BY 4.0 PAD archive.

    Existing complete extractions are reused.  Interrupted transfers remain as
    ``.part`` files and are resumed with an HTTP Range request when supported.
    """
    directory = Path(raw_root) / "PAD-UFES-20"
    directory.mkdir(parents=True, exist_ok=True)
    images, metadata = _pad_inventory(directory)
    if len(images) >= PAD_EXPECTED_IMAGES and metadata:
        return {
            "dataset": "PAD-UFES-20", "status": "ready",
            "reused_valid_images": len(images), "downloaded_bytes": 0,
            "valid_images": len(images), "metadata_files": [str(path) for path in metadata],
            "official_source_url": PAD_DOWNLOAD_URL,
        }

    archive = directory / "pad_ufes_20.zip"
    partial = directory / "pad_ufes_20.zip.part"
    if not archive.is_file():
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {
            "User-Agent": "Skin-Cancer-Classifier data preparation",
            "Referer": "https://data.mendeley.com/datasets/zr7vgbcyr2/1",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(PAD_DOWNLOAD_URL, headers=headers)
        with opener(request, timeout=120) as response:
            status = getattr(response, "status", response.getcode())
            append = bool(offset and status == 206)
            if offset and not append:
                offset = 0
            mode = "ab" if append else "wb"
            with partial.open(mode) as target:
                while True:
                    block = response.read(chunk_size)
                    if not block:
                        break
                    target.write(block)
                    if progress:
                        progress(target.tell())
        partial.replace(archive)

    _safe_extract(archive, directory)
    _extract_nested_archives(directory)
    images, metadata = _pad_inventory(directory)
    status = "ready" if len(images) >= PAD_EXPECTED_IMAGES and metadata else "incomplete"
    report = {
        "dataset": "PAD-UFES-20", "status": status,
        "reused_valid_images": 0, "downloaded_bytes": archive.stat().st_size,
        "valid_images": len(images), "metadata_files": [str(path) for path in metadata],
        "archive": str(archive), "official_source_url": PAD_DOWNLOAD_URL,
        "download_date": date.today().isoformat(),
    }
    (directory / "acquisition_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


__all__ = ["PAD_DOWNLOAD_URL", "PAD_EXPECTED_IMAGES", "download_pad_dataset"]
