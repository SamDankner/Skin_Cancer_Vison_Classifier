"""Idempotent acquisition of the official SCIN release from Google Cloud Storage."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image, UnidentifiedImageError


SCIN_BUCKET_URL = "https://storage.googleapis.com/dx-scin-public-data"
SCIN_CASES_OBJECT = "dataset/scin_cases.csv"
SCIN_LABELS_OBJECT = "dataset/scin_labels.csv"
SCIN_KNOWN_MISSING_OBJECTS = {
    "dataset/images/-2243186711511406658",
    "dataset/images/-2243186711511406658.png",
}
SCIN_IMAGE_COLUMNS = ("image_1_path", "image_2_path", "image_3_path")


def local_scin_path(directory: Path, object_name: str) -> Path:
    """Map a safe official bucket object name below the local SCIN directory."""
    relative = PurePosixPath(str(object_name).strip().lstrip("/"))
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe SCIN object path: {object_name!r}")
    return directory.joinpath(*relative.parts)


def scin_metadata_paths(directory: Path) -> tuple[Path, Path]:
    """Return official-layout paths, while accepting the legacy flat test layout."""
    official = (
        local_scin_path(directory, SCIN_CASES_OBJECT),
        local_scin_path(directory, SCIN_LABELS_OBJECT),
    )
    legacy = (directory / "scin_cases.csv", directory / "scin_labels.csv")
    return official if official[0].is_file() else legacy


def expected_scin_image_objects(cases: pd.DataFrame) -> list[str]:
    """Return unique, schema-declared image objects in stable metadata order."""
    objects: list[str] = []
    seen: set[str] = set()
    for column in SCIN_IMAGE_COLUMNS:
        if column not in cases:
            continue
        for value in cases[column].dropna():
            object_name = str(value).strip().lstrip("/")
            if object_name and object_name.lower() not in {"nan", "none", "null"} and object_name not in seen:
                local_scin_path(Path("."), object_name)
                seen.add(object_name)
                objects.append(object_name)
    return objects


def validate_image_file(path: Path) -> str | None:
    """Return an error message for a missing/corrupt image, otherwise ``None``."""
    if not path.is_file():
        return "missing"
    try:
        if path.stat().st_size == 0:
            return "zero-byte file"
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        return str(exc)
    return None


def _download_object(
    object_name: str,
    destination: Path,
    *,
    opener: Callable = urlopen,
    retries: int = 3,
) -> None:
    """Download one GCS object atomically, resuming a retained ``.part`` file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    url = f"{SCIN_BUCKET_URL}/{quote(object_name, safe='/')}"
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with opener(Request(url, headers=headers), timeout=90) as response:
                status = getattr(response, "status", response.getcode())
                append = bool(offset and status == 206)
                with partial.open("ab" if append else "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            os.replace(partial, destination)
            return
        except HTTPError:
            raise
        except (OSError, URLError, TimeoutError):
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def download_scin_dataset(
    raw_root: Path,
    *,
    opener: Callable = urlopen,
    workers: int = 16,
    progress: Callable[[str], None] | None = print,
) -> dict:
    """Download and validate the official SCIN metadata and referenced images."""
    directory = Path(raw_root) / "SCIN"
    directory.mkdir(parents=True, exist_ok=True)
    metadata_downloads = 0
    for object_name in (SCIN_CASES_OBJECT, SCIN_LABELS_OBJECT):
        destination = local_scin_path(directory, object_name)
        if not destination.is_file() or destination.stat().st_size == 0:
            _download_object(object_name, destination, opener=opener)
            metadata_downloads += 1

    cases_path, labels_path = scin_metadata_paths(directory)
    cases = pd.read_csv(cases_path, dtype={"case_id": "string"}, low_memory=False)
    objects = expected_scin_image_objects(cases)
    reusable: list[str] = []
    pending: list[str] = []
    for object_name in objects:
        destination = local_scin_path(directory, object_name)
        if validate_image_file(destination) is None:
            reusable.append(object_name)
        else:
            pending.append(object_name)

    if progress:
        progress(
            f"SCIN: {len(reusable):,} valid images already present; "
            f"{len(pending):,} official objects to fetch"
        )

    downloaded: list[str] = []
    missing: list[str] = []
    download_errors: list[dict[str, str]] = []

    def fetch(object_name: str) -> tuple[str, str | None]:
        try:
            _download_object(
                object_name,
                local_scin_path(directory, object_name),
                opener=opener,
            )
            return object_name, None
        except HTTPError as exc:
            return object_name, "missing" if exc.code == 404 else f"HTTP {exc.code}"
        except Exception as exc:  # retain partial files and continue the acquisition audit
            return object_name, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch, object_name): object_name for object_name in pending}
        completed = 0
        for future in as_completed(futures):
            object_name, error = future.result()
            completed += 1
            if error is None:
                downloaded.append(object_name)
            elif error == "missing":
                missing.append(object_name)
            else:
                download_errors.append({"object": object_name, "reason": error})
            if progress and (completed % 500 == 0 or completed == len(pending)):
                progress(f"SCIN: fetched {completed:,}/{len(pending):,} pending objects")

    corrupt: list[dict[str, str]] = []
    usable = 0
    for object_name in objects:
        error = validate_image_file(local_scin_path(directory, object_name))
        if error == "missing":
            if object_name not in missing:
                missing.append(object_name)
        elif error:
            corrupt.append({"object": object_name, "reason": error})
        else:
            usable += 1

    report = {
        "official_bucket": "dx-scin-public-data",
        "cases": len(cases),
        "expected_image_objects": len(objects),
        "metadata_downloads": metadata_downloads,
        "reused_valid_images": len(reusable),
        "downloaded_images": len(downloaded),
        "missing_images": sorted(missing),
        "known_missing_images": sorted(set(missing) & SCIN_KNOWN_MISSING_OBJECTS),
        "unexpected_missing_images": sorted(set(missing) - SCIN_KNOWN_MISSING_OBJECTS),
        "corrupt_images": corrupt,
        "download_errors": download_errors,
        "usable_images": usable,
        "metadata_files": [str(cases_path), str(labels_path)],
    }
    (directory / "acquisition_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
