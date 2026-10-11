# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import zipfile

import pytest

from lmlite import model_assets


def _write_model(root, model_id="HuggingFaceTB/SmolLM2-360M-Instruct"):
    model_dir = root.joinpath(*model_id.split("/"))
    (model_dir / "onnx").mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "onnx" / "model_q4.onnx").write_bytes(b"onnx")
    return model_dir


def _pack(source_dir, output_dir):
    return subprocess.run(
        [
            "lmlite-models",
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--format",
            "zip",
        ],
        capture_output=True,
        text=True,
    )


def test_pack_models_preserves_model_id_and_nested_files(tmp_path):
    _write_model(tmp_path)
    output_dir = tmp_path / "archives"
    result = _pack(tmp_path, output_dir)

    assert result.returncode == 0, result.stderr
    archive_path = output_dir / "model-HuggingFaceTB-SmolLM2-360M-Instruct.zip"
    assert (tmp_path / archive_path.name).read_bytes() == archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "HuggingFaceTB/SmolLM2-360M-Instruct/config.json",
            "HuggingFaceTB/SmolLM2-360M-Instruct/tokenizer.json",
            "HuggingFaceTB/SmolLM2-360M-Instruct/onnx/model_q4.onnx",
        }


def test_pack_models_packages_multiple_models(tmp_path):
    _write_model(tmp_path, "org-a/model-a")
    _write_model(tmp_path, "org-b/model-b")

    result = _pack(tmp_path, tmp_path / "archives")

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in (tmp_path / "archives").glob("*.zip")) == [
        "model-org-a-model-a.zip",
        "model-org-b-model-b.zip",
    ]


def test_pack_models_reports_byte_progress_and_preserves_large_file(tmp_path):
    model_dir = _write_model(tmp_path)
    weights = b"onnx" * (1024 * 1024)
    (model_dir / "onnx/model_q4.onnx").write_bytes(weights)

    result = _pack(tmp_path, tmp_path / "archives")

    assert result.returncode == 0, result.stderr
    assert "0.0%" in result.stderr
    assert "100.0%" in result.stderr
    assert "[##############################]" in result.stderr
    with zipfile.ZipFile(tmp_path / "archives/model-HuggingFaceTB-SmolLM2-360M-Instruct.zip") as archive:
        assert archive.read("HuggingFaceTB/SmolLM2-360M-Instruct/onnx/model_q4.onnx") == weights


def test_pack_models_excludes_hugging_face_cache_and_backup_files(tmp_path):
    model_dir = _write_model(tmp_path)
    cache_dir = model_dir / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True)
    (cache_dir / "metadata").write_text("ignored")
    (model_dir / "config.json.bak").write_text("ignored")

    result = _pack(tmp_path, tmp_path / "archives")

    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(tmp_path / "archives/model-HuggingFaceTB-SmolLM2-360M-Instruct.zip") as archive:
        names = archive.namelist()
    assert all(".cache" not in name and not name.endswith(".bak") for name in names)


def test_pack_models_fails_when_source_has_no_models(tmp_path):
    result = _pack(tmp_path, tmp_path / "archives")

    assert result.returncode == 2
    assert "no models found" in result.stderr


def test_pack_models_fails_when_source_is_missing(tmp_path):
    result = _pack(tmp_path / "missing", tmp_path / "archives")

    assert result.returncode == 2
    assert "model source directory not found" in result.stderr


def test_pack_models_reuses_source_archive_without_repacking(tmp_path):
    model_dir = _write_model(tmp_path)
    assert _pack(tmp_path, tmp_path / "first-output").returncode == 0
    archive = tmp_path / "model-HuggingFaceTB-SmolLM2-360M-Instruct.zip"
    original_bytes = archive.read_bytes()
    original_mtime = archive.stat().st_mtime_ns
    (model_dir / "config.json").write_text('{"changed": true}')

    result = _pack(tmp_path, tmp_path / "second-output")

    assert result.returncode == 0, result.stderr
    assert "reusing" in result.stdout
    assert "packing " not in result.stdout
    assert archive.read_bytes() == original_bytes
    assert archive.stat().st_mtime_ns == original_mtime
    assert (tmp_path / "second-output" / archive.name).read_bytes() == original_bytes


def test_pack_models_creates_only_missing_source_archives(tmp_path):
    _write_model(tmp_path, "org-a/model-a")
    assert _pack(tmp_path, tmp_path / "first-output").returncode == 0
    _write_model(tmp_path, "org-b/model-b")

    result = _pack(tmp_path, tmp_path / "second-output")

    assert result.returncode == 0, result.stderr
    assert "packing org-a/model-a" not in result.stdout
    assert "packing org-b/model-b" in result.stdout
    assert sorted(path.name for path in (tmp_path / "second-output").glob("*.zip")) == [
        "model-org-a-model-a.zip",
        "model-org-b-model-b.zip",
    ]


def test_pack_models_accepts_archive_only_source(tmp_path):
    archive_path = tmp_path / "model-org-model.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("org/model/config.json", "{}")

    result = _pack(tmp_path, tmp_path / "output")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output" / archive_path.name).read_bytes() == archive_path.read_bytes()


def test_pack_models_can_use_source_as_output(tmp_path):
    _write_model(tmp_path)

    first = _pack(tmp_path, tmp_path)
    second = _pack(tmp_path, tmp_path)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "reusing" in second.stdout
    assert "packing " not in second.stdout
    assert not list(tmp_path.glob(".pack-model-*"))


def test_pack_models_rejects_corrupt_cached_archive(tmp_path):
    _write_model(tmp_path)
    archive_path = tmp_path / "model-HuggingFaceTB-SmolLM2-360M-Instruct.zip"
    archive_path.write_bytes(b"incomplete ZIP")

    result = _pack(tmp_path, tmp_path / "output")

    assert result.returncode == 2
    assert "File is not a zip file" in result.stderr
    assert archive_path.read_bytes() == b"incomplete ZIP"


def test_pack_models_recreates_deleted_source_archive(tmp_path):
    model_dir = _write_model(tmp_path)
    assert _pack(tmp_path, tmp_path / "output").returncode == 0
    archive_path = tmp_path / "model-HuggingFaceTB-SmolLM2-360M-Instruct.zip"
    archive_path.unlink()
    (model_dir / "config.json").write_text('{"changed": true}')

    result = _pack(tmp_path, tmp_path / "output")

    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read("HuggingFaceTB/SmolLM2-360M-Instruct/config.json") == b'{"changed": true}'


def test_pack_models_does_not_cache_interrupted_archive(tmp_path, monkeypatch):
    _write_model(tmp_path)

    def interrupted_pack(model_id, source, output_dir):
        (output_dir / f"model-{model_id.replace('/', '-')}.zip").write_bytes(b"partial")
        raise OSError("compression interrupted")

    monkeypatch.setattr(model_assets, "pack_model", interrupted_pack)

    with pytest.raises(OSError, match="compression interrupted"):
        model_assets.pack_models(tmp_path, tmp_path / "output")

    assert not list(tmp_path.glob("model-*.zip"))
    assert not list(tmp_path.glob(".pack-model-*"))


def test_host_models_copies_individual_files_without_zip_or_download_metadata(tmp_path):
    source = tmp_path / "models"
    model = _write_model(source, "org/model")
    (model / ".cache").mkdir()
    (model / ".cache" / "metadata").write_text("excluded")
    (model / "config.json.bak").write_text("excluded")
    (model / "onnx" / "model_q4.onnx_data").write_bytes(b"external")
    _write_model(source, "org/second-model")
    output = tmp_path / "hosted"

    result = subprocess.run(
        ["lmlite-models", "--source-dir", str(source), "--output-dir", str(output), "--format", "files"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((output / "manifest.json").read_text()) == {"models": ["org/model", "org/second-model"]}
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()) == [
        "manifest.json",
        "org/model/config.json",
        "org/model/onnx/model_q4.onnx",
        "org/model/onnx/model_q4.onnx_data",
        "org/model/tokenizer.json",
        "org/second-model/config.json",
        "org/second-model/onnx/model_q4.onnx",
        "org/second-model/tokenizer.json",
    ]
    assert (output / "org/model/onnx/model_q4.onnx").read_bytes() == b"onnx"
    assert not list(source.glob("*.zip"))


def test_host_models_rejects_archive_only_source(tmp_path):
    source = tmp_path / "models"
    source.mkdir()
    with zipfile.ZipFile(source / "model-org-model.zip", "w") as archive:
        archive.writestr("org/model/config.json", "{}")

    result = subprocess.run(
        [
            "lmlite-models",
            "--source-dir",
            str(source),
            "--output-dir",
            str(tmp_path / "hosted"),
            "--format",
            "files",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "use ZIP mode" in result.stderr


def test_package_cli_defaults_to_hosted_files(tmp_path):
    source = tmp_path / "models"
    _write_model(source, "org/model")
    output = tmp_path / "hosted"

    model_assets.main(["--source-dir", str(source), "--output-dir", str(output)])

    assert json.loads((output / "manifest.json").read_text()) == {"models": ["org/model"]}
    assert (output / "org/model/onnx/model_q4.onnx").read_bytes() == b"onnx"
    assert not list(source.glob("*.zip"))


def test_package_cli_supports_explicit_zip(tmp_path):
    source = tmp_path / "models"
    _write_model(source, "org/model")
    output = tmp_path / "archives"

    model_assets.main(["--source-dir", str(source), "--output-dir", str(output), "--format", "zip"])

    with zipfile.ZipFile(output / "model-org-model.zip") as archive:
        assert archive.read("org/model/onnx/model_q4.onnx") == b"onnx"


def test_package_cli_reports_invalid_source(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        model_assets.main(["--source-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "hosted")])

    assert error.value.code == 2
    assert "model source directory not found" in capsys.readouterr().err


def test_package_cli_entry_point():
    from importlib.metadata import distribution

    entry = next(entry for entry in distribution("lmlite").entry_points if entry.name == "lmlite-models")

    assert entry.group == "console_scripts"
    assert entry.load() is model_assets.main
