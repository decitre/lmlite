# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


class TestModelBundlingOptions:
    def test_hosted_files_are_staged_for_both_builds(self, tmp_path):
        build_script = Path("scripts/lite-build.sh").read_text()
        bundle_function = build_script.split("bundle_models() {", 1)[1].split("\nprocess_for_xeus()", 1)[0]
        hosted = tmp_path / "hosted-models"
        model = hosted / "org" / "model" / "onnx"
        model.mkdir(parents=True)
        (model / "model.onnx").write_bytes(b"weights")
        (hosted / "manifest.json").write_text('{"models": ["org/model"]}')

        for kernel in ("xeus", "pyodide"):
            content = tmp_path / f"content-{kernel}"
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    "set -e\nBUNDLE_MODELS=1\n"
                    'HOSTED_MODELS_DIR="$1"\nbundle_models() {' + bundle_function + '\nbundle_models "$2"\n',
                    "stage-models",
                    str(hosted),
                    str(content),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, result.stderr
            assert (content / "models/org/model/onnx/model.onnx").read_bytes() == b"weights"
            assert (content / "models/manifest.json").read_text() == '{"models": ["org/model"]}'
            process = build_script.split(f"process_for_{kernel}() {{", 1)[1].split("\n}\n", 1)[0]
            assert process.index("bundle_models ") < process.index("jupyter lite build")
        assert "publish_models" not in build_script

    def test_zip_build_option_is_rejected(self):
        result = subprocess.run(
            ["bash", "scripts/lite-build.sh", "-z"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2
        assert "Unknown option: -z" in result.stderr

    def test_model_directories_default_to_hosted_files(self, tmp_path, monkeypatch):
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        build_script = scripts / "lite-build.sh"
        build_script.write_text(Path("scripts/lite-build.sh").read_text())
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        cli = bin_dir / "lmlite-models"
        cli.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(9)\n")
        cli.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
        source = tmp_path / "models"
        model = source / "org" / "model"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")

        result = subprocess.run(
            ["bash", str(build_script), "-M", str(source)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 9, result.stderr
        assert json.loads(result.stdout.splitlines()[-1]) == [
            "--source-dir",
            str(source),
            "--output-dir",
            str(tmp_path / "workspace" / "hosted-models"),
            "--format",
            "files",
        ]

    def test_missing_model_cli_preserves_workspace(self, tmp_path, monkeypatch):
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        build_script = scripts / "lite-build.sh"
        build_script.write_text(Path("scripts/lite-build.sh").read_text())
        source = tmp_path / "models"
        model = source / "org/model"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        marker = workspace / "keep.txt"
        marker.write_text("existing build")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        bash = shutil.which("bash")
        for command in ("dirname", "find"):
            (bin_dir / command).symlink_to(shutil.which(command))
        monkeypatch.setenv("PATH", str(bin_dir))

        result = subprocess.run(
            [bash, str(build_script), "-M", str(source)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Model hosting requires lmlite-models on PATH" in result.stderr
        assert marker.read_text() == "existing build"
        assert "Build workspace" not in result.stdout

    def test_build_without_models_does_not_require_cli(self, tmp_path, monkeypatch):
        build_script = Path("scripts/lite-build.sh").read_text()
        preflight = build_script.split('printf "${YELLOW}Build workspace', 1)[0]
        script = tmp_path / "preflight.sh"
        script.write_text(preflight)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        bash = shutil.which("bash")
        (bin_dir / "dirname").symlink_to(shutil.which("dirname"))
        monkeypatch.setenv("PATH", str(bin_dir))

        result = subprocess.run([bash, str(script)], capture_output=True, text=True)

        assert result.returncode == 0, result.stderr

    def test_archive_only_source_is_rejected(self, tmp_path):
        with zipfile.ZipFile(tmp_path / "model-org-model.zip", "w") as archive:
            archive.writestr("org/model/config.json", "{}")

        result = subprocess.run(
            ["bash", "scripts/lite-build.sh", "-M", str(tmp_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "expected <namespace>/<model>/config.json for individual-file hosting" in result.stderr

    def test_explicit_model_bundling_fails_before_build_when_source_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["bash", "scripts/lite-build.sh", "-M", tmpdir],
                capture_output=True,
                text=True,
            )

        assert result.returncode == 1
        assert "No models found" in result.stderr
