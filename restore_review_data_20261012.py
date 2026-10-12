#!/usr/bin/env python3
"""Restore compressed review inputs without changing their frozen hashes."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore(base):
    manifest_path = base / "hosting_manifest.json"
    if not manifest_path.is_file():
        print("Full review archive: processed inputs are already unpacked.")
        return
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["compressed_inputs"]:
        source = base / entry["compressed"]
        target = base / entry["original"]
        if target.is_file():
            if file_hash(target) != entry["original_sha256"]:
                raise ValueError(f"Existing input differs from frozen evidence: {entry['original']}")
            continue
        if file_hash(source) != entry["compressed_sha256"]:
            raise ValueError(f"Compressed input hash mismatch: {entry['compressed']}")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".restore-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "wb") as output, gzip.open(source, "rb") as input_stream:
                for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
            if digest.hexdigest() != entry["original_sha256"]:
                raise ValueError(f"Restored input hash mismatch: {entry['original']}")
            temporary.chmod(entry["mode"])
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"Restored {entry['original']}")


if __name__ == "__main__":
    restore(Path(__file__).resolve().parent)
