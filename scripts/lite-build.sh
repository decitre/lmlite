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
  # Every model file is hosted. MODEL_DTYPES="q8 int8" hosts only the ONNX weights of the first of these
  # dtypes each model has.
  # shellcheck disable=SC2086
  lmlite-models \
      --source-dir "$MODELS_SRC" \
      --output-dir "$HOSTED_MODELS_DIR" --format files \
      ${MODEL_DTYPES:+--dtype $MODEL_DTYPES}
fi

if command -v uv &> /dev/null; then
  pip_cmd="uv pip"
else
  pip_cmd="pip"
  $PYTHON_CMD -m pip -q install --upgrade pip
fi

# jl-forge makes breaking changes only in minor releases before 1.0 (its Compatibility Guarantees).
JL_FORGE_VERSION="~=0.5.0"
# Kept between builds: xeus environments, Pyodide release and wheel downloads, Transformers.js files.
JL_FORGE_CACHE="${JL_FORGE_CACHE:-$HOME/.cache/jl-forge}"

# Download the Transformers.js version llm.js pins, and the onnxruntime-web files it loads, for self-hosting.
TRANSFORMERS_VERSION=$(sed -n "s/^const TRANSFORMERS_VERSION = '\(.*\)';$/\1/p" "$ROOT_DIR/src/lmlite/llm.js")
TRANSFORMERS_DIR="$JL_FORGE_CACHE/lmlite-transformers/$TRANSFORMERS_VERSION"
if [ ! -f "$TRANSFORMERS_DIR/.complete" ]; then
  printf "${YELLOW}Download Transformers.js $TRANSFORMERS_VERSION${NC}\n"
  rm -rf "$TRANSFORMERS_DIR"
  mkdir -p "$TRANSFORMERS_DIR"
  npm_url="https://cdn.jsdelivr.net/npm"
  curl -fsSL -o "$TRANSFORMERS_DIR/transformers.min.js" \
      "$npm_url/@huggingface/transformers@$TRANSFORMERS_VERSION/dist/transformers.min.js"
  ort_version=$(curl -fsSL "$npm_url/@huggingface/transformers@$TRANSFORMERS_VERSION/package.json" \
      | $PYTHON_CMD -c 'import json, sys; print(json.load(sys.stdin)["dependencies"]["onnxruntime-web"])')
  # Transformers.js names the files of the onnxruntime-web version it bundles: a range could fetch another one.
  if [[ ! "$ort_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
    printf '%b\n' "${RED}onnxruntime-web dependency is not an exact version: $ort_version${NC}" >&2
    exit 1
  fi
  # asyncify: the default variant; plain: Safari < 26 without WebGPU.
  for file in ort-wasm-simd-threaded.asyncify.mjs ort-wasm-simd-threaded.asyncify.wasm \
      ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm; do
    curl -fsSL -o "$TRANSFORMERS_DIR/$file" "$npm_url/onnxruntime-web@$ort_version/dist/$file"
  done
  touch "$TRANSFORMERS_DIR/.complete"
fi

# Added to the built site, not its contents: xeus would otherwise pack them into the kernel's mounted files.
# llm.js fetches both over HTTP.
add_site_assets() {
    local site_dir="$1"
    mkdir -p "$site_dir/vendor/transformers"
    cp "$TRANSFORMERS_DIR"/* "$site_dir/vendor/transformers/"
    if [ "$BUNDLE_MODELS" = "1" ]; then
      mkdir -p "$site_dir/files/models"
      cp -R "$HOSTED_MODELS_DIR/." "$site_dir/files/models/"
    fi
}

process_for_xeus() {
    printf "${YELLOW}Create venv-xeus${NC}\n"
    pushd "$WORKSPACE"
    mkdir -p "$XEUS_WORKSPACE"
    $PYTHON_CMD -m venv venv-xeus
    source venv-xeus/bin/activate

    printf "${YELLOW}Install jupyter/xeus packages:${NC}\n"
    $pip_cmd install "jl-forge[xeus]$JL_FORGE_VERSION" jupyter_server jupyterlite-core libarchive-c

    printf "${YELLOW}Copy static content for jupyterlite-xeus site${NC}\n"
    cp "$NOTEBOOKS_DIR"/*.ipynb "$XEUS_WORKSPACE"
    ls -l "$XEUS_WORKSPACE"

    printf "${YELLOW}Create xeus-python environment${NC}\n"
    # jl-forge reuses a cached environment while the (unpinned) specs are unchanged: a cache per month picks up
    # new xeus-python and dependency releases at least monthly. Delete it to refresh sooner.
    local xeus_cache="$JL_FORGE_CACHE/xeus-$(date +%Y-%m)"
    find "$JL_FORGE_CACHE" -mindepth 1 -maxdepth 1 -name 'xeus*' ! -path "$xeus_cache" -exec rm -rf {} +
    XEUS_PREFIX=$(jl-forge xeus-env --cache-dir "$xeus_cache" --pyproject "$ROOT_DIR/pyproject.toml")

    jupyter lite build \
        --contents "$XEUS_WORKSPACE" \
        --output-dir "$XEUS_PUBLIC_DIR" \
        --XeusAddon.prefix="$XEUS_PREFIX" \
        --XeusAddon.empack_config="$(jl-forge xeus-pack-config --out "$WORKSPACE/empack.yml")" \
        --XeusAddon.mount_jupyterlite_content=True \
        --XeusAddon.mounts="$(jl-forge xeus-mount "$ROOT_DIR/src/lmlite" --prefix "$XEUS_PREFIX")"
    add_site_assets "$XEUS_PUBLIC_DIR"

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
    # Kernel 0.8.x (Python 3.14, no Voici) is a jl-forge supported combination, tested weekly against the latest release.
    $pip_cmd install build "jl-forge$JL_FORGE_VERSION" -e '..' \
      jupyter_server jupyterlab_server jupyterlite-core "jupyterlite-pyodide-kernel~=0.8.6" libarchive-c

    printf "${YELLOW}Build lmlite wheel${NC}\n"
    $PYTHON_CMD -m build "$ROOT_DIR" --wheel --outdir "$PYODIDE_WORKSPACE" --skip-dependency-check

    printf "${YELLOW}Copy static content for jupyterlite-pyodide site${NC}\n"
    cp "$NOTEBOOKS_DIR"/*.ipynb "$PYODIDE_WORKSPACE"

    printf "${YELLOW}Generate install.py and patch notebooks${NC}\n"
    jl-forge pyodide-install --pyproject "$ROOT_DIR/pyproject.toml" \
        --package-name lmlite --wheel-dir "$PYODIDE_WORKSPACE"
    jl-forge notebooks "$PYODIDE_WORKSPACE" --pyodide-install

    ls -l "$PYODIDE_WORKSPACE"

    printf "${YELLOW}Bundle PyPI wheels and self-hosted Pyodide${NC}\n"
    # jupyter lite build indexes <lite-dir>/pypi, so piplite installs these from the site.
    jl-forge pyodide-wheels --pyproject "$ROOT_DIR/pyproject.toml" \
        --cache-dir "$JL_FORGE_CACHE" --out-dir "$WORKSPACE/lite-pyodide/pypi"
    PYODIDE_DIST=$(jl-forge pyodide-dist --pyproject "$ROOT_DIR/pyproject.toml" \
        --cache-dir "$JL_FORGE_CACHE/pyodide" --out "$WORKSPACE/pyodide-dist.tar.bz2")

    jupyter lite build \
        --contents "$PYODIDE_WORKSPACE" \
        --output-dir "$PYODIDE_PUBLIC_DIR" \
        --lite-dir "$WORKSPACE/lite-pyodide" \
        --pyodide "$PYODIDE_DIST"
    add_site_assets "$PYODIDE_PUBLIC_DIR"

    printf "${YELLOW}Self-host remaining assets, dedup and check size${NC}\n"
    jl-forge self-host "$PUBLIC_DIR" --cache-dir "$JL_FORGE_CACHE/self-host"
    # dedup must run last: it hardlinks identical files, so later in-place edits would hit every copy.
    jl-forge dedup "$PUBLIC_DIR"
    # 1G is the GitHub Pages limit, which counts hardlinked files in full.
    jl-forge size "$PUBLIC_DIR" --max 1G

    jl-forge urls "$PUBLIC_DIR"
    deactivate
    popd
}

process_for_xeus
process_for_pyodide

printf "${YELLOW}Done${NC}\n"
