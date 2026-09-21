"""Safe, resumable acquisition of explicitly audited Mendeley archives.

This module deliberately acquires records only; admission to a training manifest
remains a separate, source-specific decision in :mod:`preparation`.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile


MENDELEY_RECORDS = {
    "ArsenicSkinImageBD": {
        "identifier": "x4hgnjj5gv", "version": 2,
        "record_url": "https://data.mendeley.com/datasets/x4hgnjj5gv/2",
    },
    "MCVSLD": {
        "identifier": "dfztdtfsxz", "version": 1,
        "record_url": "https://data.mendeley.com/datasets/dfztdtfsxz/1",
    },
    "MonkeyPox": {
        "identifier": "st6kggjr23", "version": 1,
        "record_url": "https://data.mendeley.com/datasets/st6kggjr23/1",
    },
    "SkinDiseaseClassification": {
        "identifier": "schhndjbjp", "version": 1,
        "record_url": "https://data.mendeley.com/datasets/schhndjbjp/1",
    },
}


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe path in Mendeley archive: {member.filename!r}")
        bundle.extractall(destination)


def _images(directory: Path) -> list[Path]:
    return [path for path in directory.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]


def download_mendeley_dataset(
    raw_root: str | Path,
    dataset: str,
    *,
    archive_name: str | None = None,
    opener=urlopen,
    chunk_size: int = 4 * 1024 * 1024,
) -> dict:
    """Download one official record without bypassing access controls.

    Interrupted files remain ``.part`` files.  A refusal is reported rather
    than retried through mirrors or alternate, unverified hosting.
    """
    if dataset not in MENDELEY_RECORDS:
        raise KeyError(f"Unknown audited Mendeley record {dataset!r}")
    record = MENDELEY_RECORDS[dataset]
    directory = Path(raw_root) / dataset
    directory.mkdir(parents=True, exist_ok=True)
    existing = _images(directory)
    if existing:
        # A prior successful call CRC-checked the official outer archive before
        # extraction.  Reuse extracted files instead of repeating a large
        # download or multi-gigabyte validation pass on every preparation run.
        previous_path = directory / "acquisition_report.json"
        previous = {}
        if previous_path.is_file():
            try:
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = {}
        report = {
            "dataset": dataset,
            "status": "downloaded_unreviewed",
            "valid_images": len(existing),
            "downloaded_bytes": 0,
            "archive": previous.get("archive"),
            "archive_crc_validated": previous.get("status") == "downloaded_unreviewed",
            "official_source_url": record["record_url"],
            "official_download_url": previous.get("official_download_url"),
            "download_date": previous.get("download_date"),
            "reused_existing_files": True,
        }
        previous_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    archive = directory / (archive_name or f"{dataset}.zip")
    partial = archive.with_suffix(".zip.part")
    download_url = f"https://data.mendeley.com/public-api/zip/{record['identifier']}/download/{record['version']}"
    try:
        if not archive.is_file():
            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {
                "User-Agent": "Skin-Cancer-Classifier data preparation",
                "Referer": record["record_url"],
            }
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(download_url, headers=headers)
            with opener(request, timeout=120) as response:
                append = bool(offset and getattr(response, "status", response.getcode()) == 206)
                with partial.open("ab" if append else "wb") as target:
                    while block := response.read(chunk_size):
                        target.write(block)
            partial.replace(archive)
        # Do not promote a partial transfer just because it has an EOCD record:
        # CRC-check every member before the atomic handoff and extraction.
        with ZipFile(archive) as bundle:
            corrupt_member = bundle.testzip()
        if corrupt_member:
            raise ValueError(f"Archive CRC validation failed for {corrupt_member!r}")
        _safe_extract(archive, directory)
        report = {
            "dataset": dataset, "status": "downloaded_unreviewed", "valid_images": len(_images(directory)),
            "downloaded_bytes": archive.stat().st_size, "archive": str(archive),
            "archive_crc_validated": True,
            "official_source_url": record["record_url"], "official_download_url": download_url,
            "download_date": date.today().isoformat(), "reused_existing_files": False,
        }
    except (HTTPError, URLError, OSError, ValueError) as exc:
        report = {
            "dataset": dataset, "status": "official_download_failed", "valid_images": len(_images(directory)),
            "downloaded_bytes": partial.stat().st_size if partial.is_file() else 0,
            "archive_crc_validated": False,
            "official_source_url": record["record_url"], "official_download_url": download_url,
            "reason": str(exc), "download_date": date.today().isoformat(),
        }
    (directory / "acquisition_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["MENDELEY_RECORDS", "download_mendeley_dataset"]
