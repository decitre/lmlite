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


class TestGenerateInstall:
    """Test generate_install.py script."""

    def test_generate_install_no_deps(self):
        """Test generating install.py with no dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal pyproject.toml
            pyproject = tmpdir / "pyproject.toml"
            pyproject.write_text(
                """
[project]
name = "test-package"
dependencies = []
"""
            )

            output = tmpdir / "install.py"
            (tmpdir / "test_package-1.2.3-py3-none-any.whl").touch()

            # Run script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_install.py",
                    "--pyproject",
                    str(pyproject),
                    "--output",
                    str(output),
                    "--package-name",
                    "test-package",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert output.exists()

            content = output.read_text()
            assert "import piplite" in content
            assert "async def _():" in content
            assert '"emfs:test_package-1.2.3-py3-none-any.whl"' in content
            assert "await piplite.install" in content

    def test_generate_install_with_deps(self):
        """Test generating install.py with dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create pyproject.toml with dependencies
            pyproject = tmpdir / "pyproject.toml"
            pyproject.write_text(
                """
[project]
name = "test-package"
dependencies = [
    "requests>=2.0",
    "numpy",
    "aiohttp",
]
"""
            )

            output = tmpdir / "install.py"
            (tmpdir / "test_package-2.3.4-py3-none-any.whl").touch()

            # Run script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_install.py",
                    "--pyproject",
                    str(pyproject),
                    "--output",
                    str(output),
                    "--package-name",
                    "test-package",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            content = output.read_text()

            # Check that dependencies are included
            assert "requests" in content
            assert "numpy" in content
            # aiohttp should be excluded
            assert "aiohttp" not in content

    def test_generate_install_uses_built_lmlite_wheel(self, tmp_path):
        wheel = tmp_path / "lmlite-0.4.0-py3-none-any.whl"
        wheel.touch()
        output = tmp_path / "install.py"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_install.py",
                "--pyproject",
                "pyproject.toml",
                "--output",
                str(output),
                "--wheel-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert 'packages = ["emfs:lmlite-0.4.0-py3-none-any.whl", "ipywidgets", "tqdm"]' in output.read_text()
        assert "0.0.0" not in output.read_text()

    def test_generate_install_rejects_missing_or_ambiguous_wheels(self, tmp_path):
        output = tmp_path / "install.py"
        command = [
            sys.executable,
            "scripts/generate_install.py",
            "--pyproject",
            "pyproject.toml",
            "--output",
            str(output),
        ]
        # Another package's wheel must not be mistaken for lmLite.
        (tmp_path / "other-1.0-py3-none-any.whl").touch()
        missing = subprocess.run(command, capture_output=True, text=True)
        assert missing.returncode == 2
        assert "found 0: none" in missing.stderr
        assert not output.exists()

        (tmp_path / "lmlite-0.3.0-py3-none-any.whl").touch()
        (tmp_path / "lmlite-0.4.0-py3-none-any.whl").touch()
        ambiguous = subprocess.run(command, capture_output=True, text=True)
        assert ambiguous.returncode == 2
        assert "found 2:" in ambiguous.stderr
        assert not output.exists()

    def test_generate_install_rejects_wheel_outside_content_directory(self, tmp_path):
        wheels = tmp_path / "wheels"
        wheels.mkdir()
        (wheels / "lmlite-0.4.0-py3-none-any.whl").touch()
        output = tmp_path / "content" / "install.py"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_install.py",
                "--pyproject",
                "pyproject.toml",
                "--output",
                str(output),
                "--wheel-dir",
                str(wheels),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2
        assert "built wheel must be alongside install.py" in result.stderr
        assert not output.exists()


class TestPatchNotebooks:
    """Test patch_notebooks.py script."""

    def test_patch_single_notebook(self):
        """Test patching a single notebook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test notebook
            notebook = tmpdir / "test.ipynb"
            nb_content = {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Title"]},
                    {"cell_type": "code", "source": ["print('hello')"]},
                    {"cell_type": "code", "source": ["print('world')"]},
                ]
            }
            notebook.write_text(json.dumps(nb_content, indent=1))

            # Run script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/patch_notebooks.py",
                    "--notebooks-dir",
                    str(tmpdir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "Patched: test.ipynb" in result.stdout

            # Check patched notebook
            patched = json.loads(notebook.read_text())
            first_code_cell = patched["cells"][1]
            assert first_code_cell["cell_type"] == "code"
            assert first_code_cell["source"][0] == "from install import _; await _()\n"
            assert "print('hello')" in first_code_cell["source"][1]

            # Second code cell should be unchanged
            second_code_cell = patched["cells"][2]
            assert second_code_cell["source"] == ["print('world')"]

    def test_patch_custom_install_line(self):
        """Test patching with custom install line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            notebook = tmpdir / "test.ipynb"
            nb_content = {
                "cells": [
                    {"cell_type": "code", "source": ["x = 1"]},
                ]
            }
            notebook.write_text(json.dumps(nb_content))

            # Run script with custom install line
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/patch_notebooks.py",
                    "--notebooks-dir",
                    str(tmpdir),
                    "--install-line",
                    "# Custom setup\n",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

            patched = json.loads(notebook.read_text())
            assert patched["cells"][0]["source"][0] == "# Custom setup\n"

    def test_skip_notebook_without_code_cells(self):
        """Test that notebooks without code cells are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            notebook = tmpdir / "markdown-only.ipynb"
            nb_content = {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Title"]},
                    {"cell_type": "markdown", "source": ["Some text"]},
                ]
            }
            notebook.write_text(json.dumps(nb_content, indent=1))

            # Run script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/patch_notebooks.py",
                    "--notebooks-dir",
                    str(tmpdir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "Skipped (no code cells)" in result.stdout

            # Notebook should be unchanged
            unchanged = json.loads(notebook.read_text())
            assert unchanged == nb_content

    def test_patch_multiple_notebooks(self):
        """Test patching multiple notebooks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create multiple notebooks
            for i in range(3):
                notebook = tmpdir / f"test{i}.ipynb"
                nb_content = {
                    "cells": [
                        {"cell_type": "code", "source": [f"print({i})"]},
                    ]
                }
                notebook.write_text(json.dumps(nb_content))

            # Run script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/patch_notebooks.py",
                    "--notebooks-dir",
                    str(tmpdir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "Patched 3 notebook(s)" in result.stdout

            # Verify all were patched
            for i in range(3):
                notebook = tmpdir / f"test{i}.ipynb"
                patched = json.loads(notebook.read_text())
                assert patched["cells"][0]["source"][0] == "from install import _; await _()\n"


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
