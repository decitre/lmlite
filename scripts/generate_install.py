#!/usr/bin/env python3
# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Generate install.py for JupyterLite pyodide kernel."""

import argparse
import re
import tomllib
from pathlib import Path


def pkg_name(dep: str) -> str:
    """Extract package name from dependency string."""
    return re.split(r"[>=<!;\[ ]", dep)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Generate install.py for pyodide")
    parser.add_argument("--pyproject", type=Path, required=True, help="Path to pyproject.toml")
    parser.add_argument("--output", type=Path, required=True, help="Output path for install.py")
    parser.add_argument("--package-name", default="lmlite", help="Package name (default: lmlite)")
    parser.add_argument("--wheel-dir", type=Path, help="Built wheel directory (default: install.py's directory)")
    args = parser.parse_args()

    # Read pyproject.toml
    with open(args.pyproject, "rb") as f:
        proj = tomllib.load(f)

    # Extract dependencies
    exclude = {"aiohttp"}  # replaced by pyfetch on emscripten
    deps = proj.get("project", {}).get("dependencies", [])
    pkgs = [pkg_name(d) for d in deps if pkg_name(d) not in exclude]

    wheel_dir = args.wheel_dir if args.wheel_dir is not None else args.output.parent
    distribution = re.sub(r"[-_.]+", "_", args.package_name).lower()
    wheels = sorted(
        path
        for path in wheel_dir.glob("*.whl")
        if path.is_file() and path.name.split("-", 1)[0].lower() == distribution
    )
    if len(wheels) != 1:
        parser.error(
            f"expected exactly one built wheel for {args.package_name} in {wheel_dir}; "
            f"found {len(wheels)}: {', '.join(path.name for path in wheels) or 'none'}"
        )
    if wheels[0].parent.resolve() != args.output.parent.resolve():
        parser.error("the built wheel must be alongside install.py for its emfs URL to resolve")

    # Build package list starting with the wheel
    all_pkgs = [f'"emfs:{wheels[0].name}"'] + [f'"{p}"' for p in pkgs]
    pkg_list = ", ".join(all_pkgs)

    # Generate install.py content
    content = (
        "import piplite\n"
        "async def _():\n"
        f"  packages = [{pkg_list}]\n"
        "  await piplite.install(packages, keep_going=True)\n"
    )

    # Write to output file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(content)

    print(f"Generated {args.output}")
    print(content)


if __name__ == "__main__":
    main()
