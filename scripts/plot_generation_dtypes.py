# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Plot generation speed by dtype and backend as a quadrant chart for docs/MODELS.md.

Data: seconds for 32 greedy tokens in Chrome 156 on an Apple M-series Mac (WASM single-threaded,
models served locally), the table in docs/MODELS.md "Generation dtype selection". Run:

    uv run scripts/plot_generation_dtypes.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TOKENS = 32
DTYPES = ["q4", "q8", "fp16", "fp32"]
SECONDS = {  # (model, backend) -> seconds per dtype; None = not measured
    ("SmolLM2-135M", "WASM"): [4.35, 0.98, 0.59, 0.57],
    ("SmolLM2-135M", "WebGPU"): [0.20, 2.39, 0.45, 0.55],
    ("SmolLM2-360M", "WASM"): [12.5, 2.25, 1.16, None],
    ("SmolLM2-360M", "WebGPU"): [0.29, 4.12, 0.74, None],
    ("TinyLlama-1.1B", "WASM"): [43.1, 5.72, None, None],
    ("TinyLlama-1.1B", "WebGPU"): [0.49, 7.58, None, None],
}
MARKERS = {"SmolLM2-135M": "o", "SmolLM2-360M": "s", "TinyLlama-1.1B": "D"}
OFFSETS = {"SmolLM2-135M": -0.14, "SmolLM2-360M": 0.0, "TinyLlama-1.1B": 0.14}  # keep markers apart
BACKEND_OFFSETS = {"WASM": -0.04, "WebGPU": 0.04}
READING = 10  # tokens/s: about twice human reading speed

THEMES = {  # validated with the dataviz palette checker (--pairs all), light and dark
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        WASM="#2a78d6",
        WebGPU="#eb6834",
        good="#e8f3e8",
        bad="#f9ece9",
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        WASM="#3987e5",
        WebGPU="#d95926",
        good="#1d2a1d",
        bad="#2e2220",
    ),
}


def plot(theme, path):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_yscale("log")
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0.5, 300)

    # Quadrant washes: top-left is the target, bottom row is too slow.
    ax.fill_between([-0.6, 1.5], READING, 300, color=t["good"], zorder=0, linewidth=0)
    ax.fill_between([-0.6, 3.6], 0.5, READING, color=t["bad"], zorder=0, linewidth=0)
    ax.axvline(1.5, color=t["axis"], linewidth=1, zorder=1)
    ax.axhline(READING, color=t["axis"], linewidth=1, zorder=1)

    corner = dict(fontsize=9, color=t["ink2"], zorder=2)
    ax.text(-0.55, 270, "Small download, fast\nbest choice", va="top", fontweight="bold", **corner)
    ax.text(3.55, 270, "Fast, larger download", va="top", ha="right", **corner)
    ax.text(1.45, 0.55, "Too slow\n(below 10 tokens/s)", va="bottom", ha="right", **corner)
    ax.text(3.55, 9, "Too slow, larger download", va="top", ha="right", **corner)

    for (model, backend), seconds in SECONDS.items():
        xs = [i + OFFSETS[model] + BACKEND_OFFSETS[backend] for i, s in enumerate(seconds) if s]
        ys = [TOKENS / s for s in seconds if s]
        ax.scatter(
            xs, ys, marker=MARKERS[model], s=64, color=t[backend], edgecolors=t["surface"], linewidths=2, zorder=4
        )

    note = dict(fontsize=8, color=t["ink2"], zorder=5)
    # Direct labels on the q4 clusters, in text ink: the legend is not the only key.
    ax.text(0.3, 64, "WebGPU\n(fails in Firefox)", va="center", **note)
    ax.text(0.3, 0.74, "WASM", va="center", **note)
    ax.text(2.6, 1.3, "TinyLlama fp16: one 2.2 GB file,\ntoo large to load in a browser", ha="center", **note)

    ax.set_xticks(range(len(DTYPES)), [f"{d}\n{b}" for d, b in zip(DTYPES, ["4-bit", "8-bit", "16-bit", "32-bit"])])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100, 200], ["0.5", "1", "2", "5", "10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("dtype (download size grows to the right)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("Generation speed (tokens/s, log scale)", color=t["ink2"], fontsize=9)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Pick the dtype for your backend: q4 on WebGPU, fp16 or q8 on WASM",
        loc="left",
        color=t["ink"],
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    handles = [Patch(color=t[b], label=b) for b in ("WebGPU", "WASM")]
    handles += [
        Line2D([], [], color=t["ink2"], marker=m, linestyle="", markersize=7, label=n) for n, m in MARKERS.items()
    ]
    legend = ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, labelcolor=t["ink2"]
    )
    legend.set_zorder(6)
    fig.text(
        0.01,
        0.01,
        "32 greedy tokens, Chrome 156, Apple M-series, WASM single-threaded. Data: docs/MODELS.md.",
        fontsize=7,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    docs = Path(__file__).resolve().parents[1] / "docs"
    plot("light", docs / "generation-dtype-speed.png")
    plot("dark", docs / "generation-dtype-speed-dark.png")
