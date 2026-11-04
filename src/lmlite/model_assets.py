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

# ONNX file-name suffix per dtype, as Transformers.js names them (onnx/<base><suffix>.onnx[_data[_N]]).
DTYPE_SUFFIXES = {
    "fp32": "",
    "fp16": "_fp16",
    "int8": "_int8",
    "uint8": "_uint8",
    "q8": "_quantized",
    "q4": "_q4",
    "q4f16": "_q4f16",
    "q2": "_q2",
    "q2f16": "_q2f16",
    "q1": "_q1",
    "q1f16": "_q1f16",
    "bnb4": "_bnb4",
}


def onnx_weight(relative_path: Path) -> tuple[str, str] | None:
    """Return ``(base, dtype)`` of an ONNX weight file (``model``, ``int8`` for model_int8.onnx), else None."""
    stem, dot, extension = relative_path.name.partition(".")
    if not dot or not (extension == "onnx" or extension.startswith("onnx_data")):
        return None
    # Longest suffix first, so "_q4f16" is not read as fp16; "" (fp32) matches last.
    for dtype, suffix in sorted(DTYPE_SUFFIXES.items(), key=lambda item: -len(item[1])):
        if stem.endswith(suffix):
            return stem[: len(stem) - len(suffix)], dtype
    return None


def onnx_dtype(relative_path: Path) -> str | None:
    """Return the dtype of an ONNX weight file, or None for any other model asset."""
    weight = onnx_weight(relative_path)
    return weight[1] if weight else None


def select_dtype(files: list[tuple[Path, Path]], dtypes: list[str]) -> list[tuple[Path, Path]]:
    """Keep the ONNX weights of the first dtype in ``dtypes`` the model has, and every non-ONNX asset."""
    weights = {weight for _, relative_path in files if (weight := onnx_weight(relative_path))}
    # Transformers.js loads onnx/model<suffix>.onnx when the model has one (decoder-only and encoder-only models);
    # their decoder_model*<suffix>.onnx files are legacy exports it never loads, so they don't make a dtype available.
    # shortcut: other components count from any file of the dtype; check the bases the model class loads if a
    # model with several components fails to load a hosted dtype.
    if any(base == "model" for base, _ in weights):
        weights = {(base, dtype) for base, dtype in weights if base == "model"}
    available = {dtype for _, dtype in weights}
    chosen = next((dtype for dtype in dtypes if dtype in available), None)
    if chosen is None:
        raise ValueError(f"none of the dtypes {', '.join(dtypes)} found; available: {', '.join(sorted(available))}")
    return [(path, rel) for path, rel in files if onnx_dtype(rel) in (None, chosen)]


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

    files, skipped = [], []
    for path in sorted(source.rglob("*")):
        relative_path = path.relative_to(source)
        if any(part in _EXCLUDED_PARTS for part in relative_path.parts) or path.suffix == ".bak":
            continue
        if path.is_symlink():
            target = path.resolve()
            # Copy link targets only from inside the model directory, so a link cannot publish unrelated files.
            if not (target.is_file() and target.is_relative_to(source.resolve())):
                print(f"skipping symlink {relative_path} -> {target}: not a file inside {source}", file=sys.stderr)
                skipped.append(relative_path.as_posix())
                continue
        if path.is_file():
            files.append((path, relative_path))

    if not any(relative_path.as_posix() == "config.json" for _, relative_path in files):
        raise ValueError(f"model directory is missing config.json: {source}")
    # A model without weights cannot load in the browser, so fail the build instead of publishing it.
    if not any(onnx_dtype(relative_path) for _, relative_path in files):
        found = ", ".join(relative_path.as_posix() for _, relative_path in files)
        raise ValueError(
            f"no ONNX weights under {source}; found: {found}; skipped symlinks: {', '.join(skipped) or 'none'}"
        )
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


def host_models(
    source_dir: Path,
    output_dir: Path,
    dtypes: list[str] | None = None,
    selected: dict[str, list[str] | None] | None = None,
) -> list[Path]:
    """Copy individual model assets and write their HTTP-discovery manifest.

    With ``dtypes``, only the ONNX weights of the first listed dtype each model has are copied.
    With ``selected`` (model id -> dtypes or None), only those models are hosted; their own dtypes replace ``dtypes``.
    """
    models = discover_models(source_dir)
    if not models:
        raise ValueError("individual-file hosting requires model directories; use ZIP mode for archive-only sources")
    if selected:
        missing = sorted(set(selected) - {model_id for model_id, _ in models})
        if missing:
            raise ValueError(f"models not found under {source_dir}: {', '.join(missing)}")
        models = [(model_id, source) for model_id, source in models if model_id in selected]
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(Path(source_dir).resolve()):
        raise ValueError("hosted output must be outside the model source directory")
    copied = []
    for model_id, source in models:
        files = model_files(source)
        model_dtypes = (selected or {}).get(model_id) or dtypes
        if model_dtypes:
            try:
                files = select_dtype(files, model_dtypes)
            except ValueError as error:
                raise ValueError(f"{model_id}: {error}") from None
        for path, relative_path in files:
            destination = output_dir / model_id / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(destination)
        hosted_dtypes = sorted({onnx_dtype(relative_path) for _, relative_path in files} - {None})
        print(f"hosting {model_id} ({', '.join(hosted_dtypes)})")
    (output_dir / "manifest.json").write_text(json.dumps({"models": [model_id for model_id, _ in models]}) + "\n")
    return copied


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--format", choices=["files", "zip"], default="files")
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=list(DTYPE_SUFFIXES),
        help="files format: host only the ONNX weights of the first of these dtypes each model has",
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="ID[:DTYPE]",
        help="files format, repeatable: host only these models, each pinned to DTYPE when given (overrides --dtype)",
    )
    args = parser.parse_args(argv)
    if (args.dtype or args.model) and args.format != "files":
        parser.error("--dtype and --model only apply to --format files")
    selected = {}
    for spec in args.model or []:
        model_id, _, dtype = spec.partition(":")
        if dtype and dtype not in DTYPE_SUFFIXES:
            parser.error(f"--model {spec}: unknown dtype {dtype!r}; choose from {', '.join(DTYPE_SUFFIXES)}")
        selected[model_id] = [dtype] if dtype else None

    try:
        if args.format == "files":
            host_models(args.source_dir, args.output_dir, args.dtype, selected)
            return
        archives = pack_models(args.source_dir, args.output_dir)
    except (NotADirectoryError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))

    for archive_path in archives:
        size_mib = archive_path.stat().st_size / (1024 * 1024)
        print(f"packed {archive_path.name} ({size_mib:.1f} MiB)")


if __name__ == "__main__":
    main()
