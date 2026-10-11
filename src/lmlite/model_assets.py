# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Prepare downloaded Hugging Face models for HTTP hosting or portable ZIP loading."""

import argparse
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

_EXCLUDED_PARTS = {".cache", ".git", "__pycache__"}


def discover_models(source_dir: Path) -> list[tuple[str, Path]]:
    """Find model directories in a ``<namespace>/<model>/config.json`` tree."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"model source directory not found: {source_dir}")

    models = []
    for config_path in sorted(source_dir.glob("*/*/config.json")):
        namespace, model = config_path.parent.parent.name, config_path.parent.name
        if namespace.startswith(".") or model.startswith("."):
            continue
        models.append((f"{namespace}/{model}", config_path.parent))

    if not models and not any(path.is_file() for path in source_dir.glob("model-*.zip")):
        raise ValueError(f"no models found under {source_dir}; expected <namespace>/<model>/config.json")
    return models


def model_files(source: Path) -> list[tuple[Path, Path]]:
    """List model assets, excluding download metadata and backup files."""
    source = Path(source)
    if not source.is_dir():
        raise NotADirectoryError(f"model directory not found: {source}")

    files = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative_path = path.relative_to(source)
        if any(part in _EXCLUDED_PARTS for part in relative_path.parts) or path.suffix == ".bak":
            continue
        files.append((path, relative_path))

    if not any(relative_path.as_posix() == "config.json" for _, relative_path in files):
        raise ValueError(f"model directory is missing config.json: {source}")
    if not files:
        raise ValueError(f"no packable model files under {source}")
    return files


def pack_model(model_id: str, source: Path, output_dir: Path) -> Path:
    """Write one model as ``model-<namespace>-<model>.zip``."""
    source = Path(source)
    files = model_files(source)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"packing {model_id} from {source} into {output_dir}")
    output_path = output_dir / f"model-{model_id.replace('/', '-')}.zip"
    total_bytes = sum(path.stat().st_size for path, _ in files)
    packed_bytes = 0
    last_update = 0.0

    def show_progress(*, finished: bool = False) -> None:
        nonlocal last_update
        now = time.monotonic()
        if not finished and now - last_update < 0.1:
            return
        last_update = now
        fraction = packed_bytes / total_bytes if total_bytes else float(finished)
        filled = min(30, int(fraction * 30))
        bar = "#" * filled + "-" * (30 - filled)
        end = "\n" if finished else "\r"
        print(
            f"\r{model_id} [{bar}] {fraction:6.1%} "
            f"({packed_bytes / (1024 * 1024):.1f}/{total_bytes / (1024 * 1024):.1f} MiB)",
            file=sys.stderr,
            end=end,
            flush=True,
        )

    show_progress()
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative_path in files:
            info = zipfile.ZipInfo.from_file(path, arcname=f"{model_id}/{relative_path.as_posix()}")
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as source_file, archive.open(info, "w", force_zip64=True) as destination:
                while chunk := source_file.read(1024 * 1024):
                    destination.write(chunk)
                    packed_bytes += len(chunk)
                    show_progress()
    show_progress(finished=True)
    return output_path


def pack_models(source_dir: Path, output_dir: Path) -> list[Path]:
    """Reuse source archives, packaging missing models before copying to output_dir."""
    source_dir = Path(source_dir)
    models = discover_models(source_dir)
    archives = {path.name: path for path in sorted(source_dir.glob("model-*.zip")) if path.is_file()}
    for archive_path in archives.values():
        # Check the ZIP directory without decompressing large model tensors.
        with zipfile.ZipFile(archive_path):
            pass
        print(f"reusing {archive_path}")

    for model_id, source in models:
        archive_name = f"model-{model_id.replace('/', '-')}.zip"
        if archive_name in archives:
            continue
        archive_path = source_dir / archive_name
        # Publish only complete archives so an interrupted build cannot leave a reusable partial ZIP.
        with tempfile.TemporaryDirectory(prefix=".pack-model-", dir=source_dir) as staging:
            packed_path = pack_model(model_id, source, Path(staging))
            packed_path.replace(archive_path)
        archives[archive_name] = archive_path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for archive_name, archive_path in sorted(archives.items()):
        destination = output_dir / archive_name
        if destination.resolve() != archive_path.resolve():
            shutil.copy2(archive_path, destination)
        result.append(destination)
    return result


def host_models(source_dir: Path, output_dir: Path) -> list[Path]:
    """Copy individual model assets and write their HTTP-discovery manifest."""
    models = discover_models(source_dir)
    if not models:
        raise ValueError("individual-file hosting requires model directories; use ZIP mode for archive-only sources")
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(Path(source_dir).resolve()):
        raise ValueError("hosted output must be outside the model source directory")
    copied = []
    for model_id, source in models:
        for path, relative_path in model_files(source):
            destination = output_dir / model_id / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(destination)
        print(f"hosting {model_id}")
    (output_dir / "manifest.json").write_text(json.dumps({"models": [model_id for model_id, _ in models]}) + "\n")
    return copied


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--format", choices=["files", "zip"], default="files")
    args = parser.parse_args(argv)

    try:
        if args.format == "files":
            host_models(args.source_dir, args.output_dir)
            return
        archives = pack_models(args.source_dir, args.output_dir)
    except (NotADirectoryError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))

    for archive_path in archives:
        size_mib = archive_path.stat().st_size / (1024 * 1024)
        print(f"packed {archive_path.name} ({size_mib:.1f} MiB)")


if __name__ == "__main__":
    main()
