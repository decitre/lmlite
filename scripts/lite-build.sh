#!/bin/bash

# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd -P)
WORKSPACE="$ROOT_DIR/workspace"
XEUS_WORKSPACE="$WORKSPACE/content-xeus"
PYODIDE_WORKSPACE="$WORKSPACE/content-pyodide"
PUBLIC_DIR="$WORKSPACE/public"
PYODIDE_PUBLIC_DIR="$PUBLIC_DIR/pyodide"
XEUS_PUBLIC_DIR="$PUBLIC_DIR/xeus"
NOTEBOOKS_DIR="$ROOT_DIR/notebooks"
TAR_CMD=$(command -v gtar || echo "tar")
PYTHON_CMD=$(command -v python3 || command -v python || echo "python3")
RED='\033[0;31m'
YELLOW='\033[33m'
NC='\033[0m'

# -m bundles models from the repository's models/ directory.
# -M bundles models from a custom directory (and implies -m).
BUNDLE_MODELS=0
MODELS_SRC="$ROOT_DIR/models"
while getopts ":mM:" opt; do
  case "$opt" in
    m) BUNDLE_MODELS=1 ;;
    M) BUNDLE_MODELS=1; MODELS_SRC="$OPTARG" ;;
    \?) printf '%b\n' "${RED}Unknown option: -$OPTARG${NC}" >&2; exit 2 ;;
    :) printf '%b\n' "${RED}Option -$OPTARG requires an argument${NC}" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [ "$#" -gt 0 ]; then
  printf '%b\n' "${RED}Unexpected argument: $1${NC}" >&2
  printf '%s\n' "usage: $0 [-m | -M models_dir]" >&2
  exit 2
fi
if [ "$BUNDLE_MODELS" = "1" ]; then
  if [[ "$MODELS_SRC" != /* ]]; then
    MODELS_SRC="$PWD/$MODELS_SRC"
  fi
  if [ ! -d "$MODELS_SRC" ]; then
    printf '%b\n' "${RED}Model bundling requested, but source directory does not exist: $MODELS_SRC${NC}" >&2
    exit 1
  fi
  MODELS_SRC=$(cd "$MODELS_SRC" && pwd -P)
  case "$MODELS_SRC/" in
    "$WORKSPACE/"*)
      printf '%b\n' "${RED}Model source cannot be inside the build workspace, which is recreated: $WORKSPACE${NC}" >&2
      exit 1
      ;;
  esac
  model_config=$(find "$MODELS_SRC" -mindepth 3 -maxdepth 3 -type f -name config.json -print -quit)
  if [ -z "$model_config" ]; then
    printf '%b\n' "${RED}No models found under $MODELS_SRC; expected <namespace>/<model>/config.json for individual-file hosting${NC}" >&2
    exit 1
  fi
  if ! command -v lmlite-models &> /dev/null; then
    printf '%b\n' "${RED}Model hosting requires lmlite-models on PATH. Install lmLite in your build environment.${NC}" >&2
    exit 1
  fi
fi

printf "${YELLOW}Build workspace${NC}\n"
rm -rf "$WORKSPACE"
mkdir -p "$WORKSPACE/content-xeus" "$WORKSPACE/content-pyodide"

HOSTED_MODELS_DIR="$WORKSPACE/hosted-models"
if [ "$BUNDLE_MODELS" = "1" ]; then
  printf "${YELLOW}Prepare hosted model files${NC}\n"
  lmlite-models \
      --source-dir "$MODELS_SRC" \
      --output-dir "$HOSTED_MODELS_DIR" --format files
fi

if command -v uv &> /dev/null; then
  pip_cmd="uv pip"
else
  pip_cmd="pip"
  $PYTHON_CMD -m pip -q install --upgrade pip
fi

bundle_models() {
    local content_dir="$1"
    [ "$BUNDLE_MODELS" = "1" ] || return 0
    mkdir -p "$content_dir/models"
    cp -R "$HOSTED_MODELS_DIR/." "$content_dir/models/"
}

process_for_xeus() {
    printf "${YELLOW}Create venv-xeus${NC}\n"
    pushd "$WORKSPACE"
    mkdir -p "$XEUS_WORKSPACE"
    $PYTHON_CMD -m venv venv-xeus
    source venv-xeus/bin/activate

    printf "${YELLOW}Install jupyter/xeus packages:${NC}\n"
    $pip_cmd install jupyter_server jupyterlite-core jupyterlite-xeus libarchive-c

    printf "${YELLOW}Copy static content for jupyterlite-xeus site${NC}\n"
    cp "$NOTEBOOKS_DIR"/*.ipynb "$XEUS_WORKSPACE"
    bundle_models "$XEUS_WORKSPACE"
    ls -l "$XEUS_WORKSPACE"

    cat << eof > environment.yml
name: xeus-kernels
channels:
  - https://prefix.dev/emscripten-forge-dev
  - https://prefix.dev/conda-forge
dependencies:
  - xeus-python
  - tqdm
  - ipywidgets

eof

    jupyter lite build \
        --contents "$XEUS_WORKSPACE" \
        --output-dir "$XEUS_PUBLIC_DIR" \
        --XeusAddon.environment_file=$(pwd)/environment.yml \
        --XeusAddon.mount_jupyterlite_content=True \
        --XeusAddon.mounts="$ROOT_DIR/src/lmlite:/lib/python3.13/site-packages/lmlite"

    deactivate
    popd
}

process_for_pyodide() {
    printf "${YELLOW}Create venv-pyodide${NC}\n"
    pushd "$WORKSPACE"
    mkdir -p "$PYODIDE_WORKSPACE"
    $PYTHON_CMD -m venv venv-pyodide
    source venv-pyodide/bin/activate

    printf "${YELLOW}Install jupyter/pyodide packages:${NC}\n"
    # https://pyodide.org/en/stable/usage/packages-in-pyodide.html
    $pip_cmd install build -e '..' \
      jupyter_server jupyterlab_server jupyterlite-core jupyterlite-pyodide-kernel libarchive-c

    printf "${YELLOW}Build lmlite wheel${NC}\n"
    $PYTHON_CMD -m build "$ROOT_DIR" --wheel --outdir "$PYODIDE_WORKSPACE" --skip-dependency-check

    printf "${YELLOW}Copy static content for jupyterlite-pyodide site${NC}\n"
    cp "$NOTEBOOKS_DIR"/*.ipynb "$PYODIDE_WORKSPACE"

    printf "${YELLOW}Generate install.py${NC}\n"
    $PYTHON_CMD "$SCRIPT_DIR/generate_install.py" \
        --pyproject "$ROOT_DIR/pyproject.toml" \
        --output "$PYODIDE_WORKSPACE/install.py" \
        --wheel-dir "$PYODIDE_WORKSPACE" \
        --package-name lmlite

    printf "${YELLOW}Patch notebooks${NC}\n"
    $PYTHON_CMD "$SCRIPT_DIR/patch_notebooks.py" \
        --notebooks-dir "$PYODIDE_WORKSPACE"

    bundle_models "$PYODIDE_WORKSPACE"
    ls -l "$PYODIDE_WORKSPACE"

    jupyter lite build \
        --contents "$PYODIDE_WORKSPACE" \
        --output-dir "$PYODIDE_PUBLIC_DIR" \
        --lite-dir "$PYODIDE_PUBLIC_DIR"

    deactivate
    popd
}

process_for_xeus
process_for_pyodide

printf "${YELLOW}Done${NC}\n"

echo ""
echo "Serve locally:"
echo "python -m http.server 8000 --directory $PUBLIC_DIR 2>/dev/null"
echo ""
echo "JupyterLab Xeus UI:"
echo "http://localhost:8000/xeus/"
echo ""
echo "JupyterLab Pyodide UI:"
echo "http://localhost:8000/pyodide/"
echo ""
