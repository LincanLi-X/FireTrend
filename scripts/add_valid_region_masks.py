#!/usr/bin/env python3
"""Add coordinate-aligned U.S. state masks to the FireCast HDF5 subsets.

The script preserves every pre-existing HDF5 dataset, creates a recoverable
pre-mask backup, writes through a temporary HDF5 copy, validates the copy, and
then atomically replaces the working file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone
from urllib.request import urlopen
from zipfile import ZipFile

import h5py
import numpy as np
import shapefile
from shapely.geometry import Point, shape
from shapely.validation import make_valid


BOUNDARY_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2025/shp/"
    "cb_2025_us_state_500k.zip"
)
BOUNDARY_SHA256 = "9cbfe171dad1555e11770c981d8f4db9e687a65c86f5bdae684eeb487e2e9b80"
BOUNDARY_FILENAME = "cb_2025_us_state_500k.zip"
MASK_DATASET = "valid_region_mask"
IGNORE_INDEX = -100

EXPECTED_FILES = {
    "CA_wildfire_grid_ERA5_LANDFIRE_aligned_subset.h5": "CA",
    "FL_wildfire_grid_ERA5_LANDFIRE_aligned_subset.h5": "FL",
    "OR_wildfire_grid_ERA5_LANDFIRE_aligned.h5": "OR",
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_dataset(dataset: h5py.Dataset) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset.dtype).encode("utf-8"))
    digest.update(np.asarray(dataset.shape, dtype=np.int64).tobytes())
    if dataset.ndim == 0:
        digest.update(np.ascontiguousarray(dataset[()]).tobytes())
        return digest.hexdigest()
    if dataset.shape[0] == 0:
        return digest.hexdigest()
    for start in range(0, dataset.shape[0], 32):
        stop = min(start + 32, dataset.shape[0])
        digest.update(np.ascontiguousarray(dataset[start:stop]).tobytes())
    return digest.hexdigest()


def scientific_dataset_manifest(path: Path) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    with h5py.File(path, "r") as h5:
        for name in sorted(h5.keys()):
            if name == MASK_DATASET:
                continue
            dataset = h5[name]
            if not isinstance(dataset, h5py.Dataset):
                continue
            manifest[name] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "sha256": sha256_dataset(dataset),
            }
    return manifest


def download_boundary_archive(cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        actual = sha256_file(cache_path)
        if actual != BOUNDARY_SHA256:
            raise RuntimeError(
                f"Cached boundary checksum mismatch: expected {BOUNDARY_SHA256}, got {actual}"
            )
        return cache_path

    temporary = cache_path.with_suffix(cache_path.suffix + ".download")
    try:
        with urlopen(BOUNDARY_URL, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = sha256_file(temporary)
        if actual != BOUNDARY_SHA256:
            raise RuntimeError(
                f"Downloaded boundary checksum mismatch: expected {BOUNDARY_SHA256}, got {actual}"
            )
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return cache_path


def load_state_geometries(boundary_archive: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="firecast-boundaries-") as work:
        with ZipFile(boundary_archive) as archive:
            archive.extractall(work)
        shapefiles = list(Path(work).glob("*.shp"))
        if len(shapefiles) != 1:
            raise RuntimeError(f"Expected one shapefile, found {len(shapefiles)}")
        reader = shapefile.Reader(str(shapefiles[0]))
        fields = [field[0] for field in reader.fields[1:]]
        if "STUSPS" not in fields:
            raise RuntimeError("Boundary shapefile has no STUSPS field")
        state_index = fields.index("STUSPS")
        geometries: dict[str, object] = {}
        for item in reader.iterShapeRecords():
            state_code = item.record[state_index]
            geometry = shape(item.shape.__geo_interface__)
            if not geometry.is_valid:
                geometry = make_valid(geometry)
            geometries[state_code] = geometry
    missing = sorted(set(EXPECTED_FILES.values()) - set(geometries))
    if missing:
        raise RuntimeError(f"Missing state geometries: {missing}")
    return geometries


def preflight_hdf5(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    with h5py.File(path, "r") as h5:
        required = {"wildfire_risk", "latitude", "longitude"}
        missing = sorted(required - set(h5.keys()))
        if missing:
            raise RuntimeError(f"{path.name} is missing datasets: {missing}")
        risk_shape = tuple(h5["wildfire_risk"].shape)
        latitude = np.asarray(h5["latitude"][:], dtype=np.float64)
        longitude = np.asarray(h5["longitude"][:], dtype=np.float64)

    if len(risk_shape) != 3:
        raise RuntimeError(f"{path.name}: wildfire_risk must be (T,H,W), got {risk_shape}")
    if latitude.shape != (risk_shape[1],) or longitude.shape != (risk_shape[2],):
        raise RuntimeError(
            f"{path.name}: coordinate shapes do not match wildfire_risk: "
            f"lat={latitude.shape}, lon={longitude.shape}, risk={risk_shape}"
        )
    if not np.all(np.diff(latitude) < 0):
        raise RuntimeError(f"{path.name}: latitude must be strictly descending")
    if not np.all(np.diff(longitude) > 0):
        raise RuntimeError(f"{path.name}: longitude must be strictly ascending")
    if not np.allclose(np.diff(latitude), -0.25, atol=1e-8):
        raise RuntimeError(f"{path.name}: latitude spacing is not -0.25 degrees")
    if not np.allclose(np.diff(longitude), 0.25, atol=1e-8):
        raise RuntimeError(f"{path.name}: longitude spacing is not 0.25 degrees")
    return latitude, longitude, risk_shape


def build_mask(latitude: np.ndarray, longitude: np.ndarray, geometry: object) -> np.ndarray:
    mask = np.zeros((latitude.size, longitude.size), dtype=np.uint8)
    for row, lat in enumerate(latitude):
        for column, lon in enumerate(longitude):
            mask[row, column] = int(geometry.covers(Point(float(lon), float(lat))))
    unique = set(np.unique(mask).tolist())
    if not unique.issubset({0, 1}):
        raise RuntimeError(f"Mask contains unexpected values: {sorted(unique)}")
    if int(mask.sum()) == 0 or int(mask.sum()) == mask.size:
        raise RuntimeError(f"Mask is degenerate: valid={int(mask.sum())}, total={mask.size}")
    return mask


def create_backup(source: Path, backup_dir: Path) -> tuple[Path, str, str]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / source.name
    source_checksum = sha256_file(source)
    if backup.exists():
        backup_checksum = sha256_file(backup)
        return backup, backup_checksum, "existing"
    try:
        os.link(source, backup)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, backup)
        method = "copy"
    backup_checksum = sha256_file(backup)
    if backup_checksum != source_checksum:
        raise RuntimeError(f"Backup verification failed for {source.name}")
    return backup, backup_checksum, method


def write_mask_atomically(
    source: Path,
    state_code: str,
    mask: np.ndarray,
    created_utc: str,
    original_manifest: dict[str, dict[str, object]],
) -> tuple[str, str]:
    temporary = source.with_name(f".{source.name}.mask-update.tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    try:
        with h5py.File(temporary, "r+") as h5:
            if MASK_DATASET in h5:
                del h5[MASK_DATASET]
            dataset = h5.create_dataset(
                MASK_DATASET,
                data=mask,
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            dataset.attrs["long_name"] = "Static valid target-state land-region mask"
            dataset.attrs["flag_values"] = np.asarray([0, 1], dtype=np.uint8)
            dataset.attrs["flag_meanings"] = "invalid valid"
            dataset.attrs["state_code"] = state_code
            dataset.attrs["crs"] = "EPSG:4326"
            dataset.attrs["boundary_source"] = BOUNDARY_URL
            dataset.attrs["boundary_vintage"] = np.int32(2025)
            dataset.attrs["boundary_scale"] = "1:500000"
            dataset.attrs["boundary_sha256"] = BOUNDARY_SHA256
            dataset.attrs["rasterization_rule"] = "pixel_center"
            dataset.attrs["all_touched"] = np.uint8(0)
            dataset.attrs["invalid_class_label"] = np.int32(IGNORE_INDEX)
            dataset.attrs["created_utc"] = created_utc
            dataset.attrs["mask_sha256"] = sha256_array(mask)

            valid = int(mask.sum())
            invalid = int(mask.size - valid)
            h5.attrs["valid_region_mask_dataset"] = MASK_DATASET
            h5.attrs["valid_grid_regions"] = np.int64(valid)
            h5.attrs["invalid_grid_regions"] = np.int64(invalid)
            h5.attrs["valid_grid_fraction"] = np.float64(valid / mask.size)
            h5.attrs["invalid_grid_fraction"] = np.float64(invalid / mask.size)
            h5.attrs["invalid_class_label"] = np.int32(IGNORE_INDEX)
            h5.flush()

        updated_manifest = scientific_dataset_manifest(temporary)
        if updated_manifest != original_manifest:
            raise RuntimeError(f"A pre-existing scientific dataset changed in {source.name}")
        with h5py.File(temporary, "r") as h5:
            written = np.asarray(h5[MASK_DATASET][:], dtype=np.uint8)
            if not np.array_equal(written, mask):
                raise RuntimeError(f"Written mask verification failed for {source.name}")
            if h5[MASK_DATASET].shape != mask.shape:
                raise RuntimeError(f"Written mask shape verification failed for {source.name}")

        before_replace = sha256_file(source)
        os.replace(temporary, source)
        after_replace = sha256_file(source)
        return before_replace, after_replace
    finally:
        if temporary.exists():
            temporary.unlink()


def build_audit_entry(
    path: Path,
    state_code: str,
    mask: np.ndarray,
    backup_path: Path,
    backup_sha256: str,
    backup_method: str,
    file_sha256_before: str,
    file_sha256_after: str,
) -> dict[str, object]:
    with h5py.File(path, "r") as h5:
        risk = np.asarray(h5["wildfire_risk"][:], dtype=np.float32)
        positive = risk > 0
        positive_total = int(positive.sum())
        positive_outside = int((positive & (mask[None, :, :] == 0)).sum())
        ever_positive = np.any(positive, axis=0)
        entry = {
            "state_code": state_code,
            "file": path.name,
            "file_sha256_before_update": file_sha256_before,
            "file_sha256_after_update": file_sha256_after,
            "backup_file": str(backup_path.relative_to(path.parent)),
            "backup_sha256": backup_sha256,
            "backup_method": backup_method,
            "spatial_shape": list(mask.shape),
            "time_steps": int(risk.shape[0]),
            "valid_pixels": int(mask.sum()),
            "invalid_pixels": int(mask.size - mask.sum()),
            "valid_fraction": float(mask.mean()),
            "invalid_fraction": float(1.0 - mask.mean()),
            "mask_sha256": sha256_array(mask),
            "positive_risk_pixel_days": positive_total,
            "positive_risk_pixel_days_outside_mask": positive_outside,
            "positive_risk_pixel_days_outside_fraction": (
                float(positive_outside / positive_total) if positive_total else 0.0
            ),
            "ever_positive_cells_inside_mask": int((ever_positive & (mask == 1)).sum()),
            "ever_positive_cells_outside_mask": int((ever_positive & (mask == 0)).sum()),
            "scientific_datasets_unchanged": True,
        }
    return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_v2",
        help="Directory containing the FireCast HDF5 files.",
    )
    parser.add_argument(
        "--boundary-archive",
        type=Path,
        default=None,
        help="Optional path to the pinned Census boundary ZIP.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report masks without updating HDF5 files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    boundary_archive = (
        args.boundary_archive.resolve()
        if args.boundary_archive
        else data_dir / "boundary_sources" / BOUNDARY_FILENAME
    )
    if args.boundary_archive:
        actual_boundary_checksum = sha256_file(boundary_archive)
        if actual_boundary_checksum != BOUNDARY_SHA256:
            raise RuntimeError(
                f"Boundary checksum mismatch: expected {BOUNDARY_SHA256}, "
                f"got {actual_boundary_checksum}"
            )
    else:
        boundary_archive = download_boundary_archive(boundary_archive)

    geometries = load_state_geometries(boundary_archive)
    created_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    audit: dict[str, object] = {
        "schema_version": 1,
        "created_utc": created_utc,
        "mask_dataset": MASK_DATASET,
        "mask_values": {"0": "invalid", "1": "valid"},
        "invalid_class_label": IGNORE_INDEX,
        "boundary_source": BOUNDARY_URL,
        "boundary_sha256": BOUNDARY_SHA256,
        "boundary_vintage": 2025,
        "boundary_scale": "1:500000",
        "crs": "EPSG:4326",
        "rasterization_rule": "pixel_center",
        "all_touched": False,
        "files": [],
    }

    for filename, state_code in EXPECTED_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        latitude, longitude, risk_shape = preflight_hdf5(path)
        original_manifest = scientific_dataset_manifest(path)
        mask = build_mask(latitude, longitude, geometries[state_code])
        if mask.shape != risk_shape[1:]:
            raise RuntimeError(f"{filename}: mask shape {mask.shape} != {risk_shape[1:]}")

        if args.dry_run:
            print(
                f"{state_code}: shape={mask.shape}, valid={int(mask.sum())}, "
                f"invalid={int(mask.size - mask.sum())}, mask_sha256={sha256_array(mask)}"
            )
            continue

        backup_path, backup_sha256, backup_method = create_backup(
            path, data_dir / "backups_pre_valid_region_mask"
        )
        before, after = write_mask_atomically(
            path,
            state_code,
            mask,
            created_utc,
            original_manifest,
        )
        audit["files"].append(
            build_audit_entry(
                path,
                state_code,
                mask,
                backup_path,
                backup_sha256,
                backup_method,
                before,
                after,
            )
        )
        print(
            f"Updated {filename}: valid={int(mask.sum())}, "
            f"invalid={int(mask.size - mask.sum())}"
        )

    if not args.dry_run:
        audit_path = data_dir / "valid_region_mask_audit.json"
        temporary_audit = audit_path.with_suffix(".json.tmp")
        temporary_audit.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_audit, audit_path)
        print(f"Wrote audit report: {audit_path}")


if __name__ == "__main__":
    main()
