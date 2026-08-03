#!/usr/bin/env python3
"""Validate FireCast masks, backups, provenance, and untouched source arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from add_valid_region_masks import (
    BOUNDARY_FILENAME,
    BOUNDARY_SHA256,
    EXPECTED_FILES,
    MASK_DATASET,
    build_mask,
    load_state_geometries,
    preflight_hdf5,
    scientific_dataset_manifest,
    sha256_array,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_v2",
    )
    return parser.parse_args()


def main() -> None:
    data_dir = parse_args().data_dir.resolve()
    archive = data_dir / "boundary_sources" / BOUNDARY_FILENAME
    if sha256_file(archive) != BOUNDARY_SHA256:
        raise RuntimeError("Pinned Census boundary archive checksum does not match")
    geometries = load_state_geometries(archive)

    audit_path = data_dir / "valid_region_mask_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_by_file = {entry["file"]: entry for entry in audit["files"]}

    for filename, state_code in EXPECTED_FILES.items():
        path = data_dir / filename
        backup = data_dir / "backups_pre_valid_region_mask" / filename
        entry = audit_by_file[filename]
        if sha256_file(backup) != entry["backup_sha256"]:
            raise RuntimeError(f"Backup checksum mismatch for {filename}")
        if scientific_dataset_manifest(path) != scientific_dataset_manifest(backup):
            raise RuntimeError(f"Pre-existing dataset mismatch for {filename}")

        latitude, longitude, risk_shape = preflight_hdf5(path)
        expected_mask = build_mask(latitude, longitude, geometries[state_code])
        with h5py.File(path, "r") as h5:
            if MASK_DATASET not in h5:
                raise RuntimeError(f"Missing {MASK_DATASET} in {filename}")
            actual_mask = np.asarray(h5[MASK_DATASET][:], dtype=np.uint8)
            if actual_mask.shape != risk_shape[1:]:
                raise RuntimeError(f"Mask shape mismatch for {filename}")
            if not np.array_equal(actual_mask, expected_mask):
                raise RuntimeError(f"Mask does not match Census rasterization for {filename}")
            if sha256_array(actual_mask) != entry["mask_sha256"]:
                raise RuntimeError(f"Mask checksum mismatch for {filename}")
            if int(h5.attrs["valid_grid_regions"]) != int(actual_mask.sum()):
                raise RuntimeError(f"Valid-pixel metadata mismatch for {filename}")
            if int(h5.attrs["invalid_grid_regions"]) != int(actual_mask.size - actual_mask.sum()):
                raise RuntimeError(f"Invalid-pixel metadata mismatch for {filename}")
            if int(h5.attrs["invalid_class_label"]) != -100:
                raise RuntimeError(f"Ignore-label metadata mismatch for {filename}")
        print(
            f"PASS {state_code}: valid={int(actual_mask.sum())}, "
            f"invalid={int(actual_mask.size - actual_mask.sum())}, "
            f"mask_sha256={sha256_array(actual_mask)}"
        )

    print("All FireCast valid-region mask acceptance checks passed.")


if __name__ == "__main__":
    main()
