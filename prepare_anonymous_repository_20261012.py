#!/usr/bin/env python3
"""Prepare a complete review repository under the hosting file-size limit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


FILE_LIMIT = 7_900_000


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, destination, archive_path):
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists() or destination == source or destination in source.parents or source in destination.parents:
        raise ValueError("Repository output must be a new directory separate from the review artifact")
    original_manifest = json.loads((source / "manifest.json").read_text())
    for name, expected in original_manifest["sha256"].items():
        if file_hash(source / name) != expected:
            raise ValueError(f"Review artifact hash mismatch: {name}")
    destination.mkdir(parents=True)
    compressed = []
    for original in sorted(source.rglob("*")):
        if not original.is_file():
            continue
        name = original.relative_to(source).as_posix()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.stat().st_size >= FILE_LIMIT:
            target = target.with_name(target.name + ".gz")
            with original.open("rb") as input_stream, target.open("wb") as compressed_stream:
                with gzip.GzipFile(filename="", mode="wb", fileobj=compressed_stream, compresslevel=6, mtime=0) as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
            if target.stat().st_size >= FILE_LIMIT:
                raise ValueError(f"Compressed input still exceeds the hosting limit: {name}")
            compressed.append({"original": name, "original_sha256": file_hash(original), "compressed": target.relative_to(destination).as_posix(), "compressed_sha256": file_hash(target), "mode": original.stat().st_mode & 0o777})
        else:
            shutil.copy2(original, target)
    ignored = ["work/venv/", "__pycache__/", "*.pyc", "/output/xeon-*/", "/output/artifacts/", "/output/anonymous_repository/", "/output/*.zip"]
    ignored.extend("/" + entry["original"] for entry in compressed)
    (destination / ".gitignore").write_text("\n".join(ignored) + "\n")
    manifest = {
        "status": "prepared for anonymous hosting; not uploaded",
        "hosting_service": "https://anonymous.4open.science/",
        "source_visibility": "use a private source repository; no redirect to source after expiration",
        "file_size_limit_bytes": FILE_LIMIT,
        "review_manifest_sha256": file_hash(destination / "manifest.json"),
        "compressed_inputs": compressed,
        "restore_command": "python restore_review_data_20261012.py",
        "sha256": {path.relative_to(destination).as_posix(): file_hash(path) for path in sorted(destination.rglob("*")) if path.is_file()},
    }
    (destination / "hosting_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if any(path.stat().st_size >= FILE_LIMIT for path in destination.rglob("*") if path.is_file()):
        raise ValueError("A generated file exceeds the hosting limit")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(destination.name) / path.relative_to(destination)).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Repository archive CRC failed")
    print(json.dumps({"directory": str(destination), "archive": str(archive_path), "archive_bytes": archive_path.stat().st_size, "archive_sha256": file_hash(archive_path), "compressed_inputs": len(compressed), "files": len(manifest["sha256"]) + 1, "largest_file_bytes": max(path.stat().st_size for path in destination.rglob("*") if path.is_file())}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output_dir, args.archive)


if __name__ == "__main__":
    main()
