"""Helpers shared by the dataset generators: the raw file check, feature standardization and the dataset files."""

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.files import sha256


def check_raw_file(path, size, digest):
    """A raw source file has to be the one the reported datasets were generated from."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"{path} is missing; the Data section of the README says where to download it")
    if (path.stat().st_size, sha256(path)) != (size, digest):
        raise SystemExit(
            f"{path} is not the expected file: it has {path.stat().st_size} bytes, expected are {size} bytes with "
            f"sha256 {digest}"
        )


def standardize_features(X, training_count):
    """Features standardized with the mean and scale of the TRAIN rows: (float32 X, {mean, scale} of each feature)."""
    feature_dim = X.shape[-1]
    scaler = StandardScaler().fit(X[:training_count].reshape(-1, feature_dim))
    transformed = scaler.transform(X.reshape(-1, feature_dim)).reshape(X.shape)
    return transformed.astype(np.float32), {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def save_dataset(directory, arrays, metadata):
    """Write <name>.npy for every array, and meta.json: the metadata with the shape, dtype and sha256 of each array."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(directory / f"{name}.npy", value)
    files = {
        f"{name}.npy": {"shape": list(v.shape), "dtype": v.dtype.name, "sha256": sha256(directory / f"{name}.npy")}
        for name, v in arrays.items()
    }
    (directory / "meta.json").write_text(json.dumps(metadata | {"files": files}, indent=1) + "\n")
