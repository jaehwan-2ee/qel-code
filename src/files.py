"""File helpers: the sha256 of a file, and JSON records."""

import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    # allow_nan=False: a record never holds NaN or Infinity, which are not JSON
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
