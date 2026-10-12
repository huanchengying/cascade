#!/usr/bin/env python3
"""Generate paper figures from the checked-in temporal-DAG benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d9dee5",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
    }
)

COLORS = {
    "huang": "#4C78A8",
    "tegraph": "#F58518",
    "legacy": "#B279A2",
    "compact": "#2A9D8F",
    "build": "#8FBBD9",
    "solve": "#2A9D8F",
}


def load_shortest() -> dict[tuple[int, str], dict[str, float]]:
    with (ROOT / "benchmark_shortest_path.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    grouped: dict[tuple[int, str], list[dict[str, float]]] = {}
    for row in payload["records"]:
        if row.get("status") != "ok":
            continue
        key = (int(row["prefix_edges"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    # The benchmark has three isolated repetitions.  Use the median reported
    # in the paper tables rather than selecting one potentially noisy run.
    result: dict[tuple[int, str], dict[str, float]] = {}
    for key, rows in grouped.items():
        result[key] = {
            field: float(np.median([float(row[field]) for row in rows]))
            for field in ("end_to_end_seconds", "build_seconds", "solve_seconds", "peak_rss_mib")
        }
    return result


def load_cross_application() -> dict[tuple[int, str], float]:
    result: dict[tuple[int, str], float] = {}
    with (ROOT / "benchmark_comparison.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if not method.startswith("compact_"):
                continue
            app = method.removeprefix("compact_").replace("_upper_bound", "")
            result[(int(row["prefix_edges"]), app)] = float(row["end_to_end_seconds"])
    return result


def annotate_bars(ax: plt.Axes, bars: list[object], values: list[float], fmt: str = "{:.2f}") -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
            clip_on=False,
        )


def shortest_end_to_end(rows: dict[tuple[int, str], dict[str, float]]) -> None:
    names = {
        "Huang_style_state_DAG_shortest": "Huang\nstate DAG",
        "TeGraph_style_in_out_DAG_shortest": "TeGraph\nin/out DAG",
        "Proposed_lazy_hyper_DAG_shortest": "Legacy\nlazy",
        "Compact_DAG_shortest": "Compact\nTDAG",
    }
    order = list(names)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=False)
    for ax, prefix in zip(axes, (100_000, 1_000_000)):
        methods = [method for method in order if (prefix, method) in rows]
        values = [rows[(prefix, method)]["end_to_end_seconds"] for method in methods]
        colors = [
            COLORS["huang"] if "Huang" in method else
            COLORS["tegraph"] if "TeGraph" in method else
            COLORS["legacy"] if "Proposed" in method else COLORS["compact"]
            for method in methods
        ]
        bars = ax.bar(np.arange(len(methods)), values, color=colors, width=0.68, edgecolor="white", linewidth=0.5)
        annotate_bars(ax, list(bars), values)
        ax.set_xticks(np.arange(len(methods)), [names[method] for method in methods])
        ax.set_ylabel("End-to-end time (s)")
        ax.set_title(f"{prefix // 1000}K-edge prefix")
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_axisbelow(True)
    fig.suptitle("Temporal shortest-path latency", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_shortest_end_to_end.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_shortest_end_to_end.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def shortest_breakdown(rows: dict[tuple[int, str], dict[str, float]]) -> None:
    names = {
        "Huang_style_state_DAG_shortest": "Huang",
        "TeGraph_style_in_out_DAG_shortest": "TeGraph",
        "Proposed_lazy_hyper_DAG_shortest": "Legacy lazy",
        "Compact_DAG_shortest": "Compact TDAG",
    }
    order = list(names)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=False)
    for ax, prefix in zip(axes, (100_000, 1_000_000)):
        methods = [method for method in order if (prefix, method) in rows]
        x = np.arange(len(methods))
        build = [rows[(prefix, method)]["build_seconds"] for method in methods]
        solve = [rows[(prefix, method)]["solve_seconds"] for method in methods]
        ax.bar(x, build, color=COLORS["build"], width=0.62, label="Build")
        ax.bar(x, solve, bottom=build, color=COLORS["solve"], width=0.62, label="Solve")
        totals = [a + b for a, b in zip(build, solve)]
        for xi, total in zip(x, totals):
            ax.text(xi, total, f"{total:.2f}", ha="center", va="bottom", fontsize=7, clip_on=False)
        ax.set_xticks(x, [names[method] for method in methods], rotation=18, ha="right")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"{prefix // 1000}K-edge prefix")
        ax.set_ylim(0, max(totals) * 1.27)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.suptitle("Construction dominates end-to-end shortest-path cost", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_shortest_breakdown.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_shortest_breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def cross_application(rows: dict[tuple[int, str], float]) -> None:
    apps = ["reachability", "mst_a", "shortest", "fastest", "mst_w"]
    labels = ["Reachability", r"$\mathrm{MST}_a$", "Shortest", "Fastest", r"$\mathrm{MST}_w$\n(upper bound)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharey=False)
    for ax, prefix in zip(axes, (100_000, 1_000_000)):
        values = [rows[(prefix, app)] for app in apps]
        bars = ax.bar(np.arange(len(apps)), values, color=[COLORS["compact"]] * len(apps), width=0.65)
        annotate_bars(ax, list(bars), values)
        ax.set_xticks(np.arange(len(apps)), labels, rotation=18, ha="right")
        ax.set_ylabel("End-to-end time (s)")
        ax.set_title(f"{prefix // 1000}K-edge prefix")
        ax.set_ylim(0, max(values) * 1.27)
        ax.set_axisbelow(True)
    fig.suptitle("One compact index, multiple temporal applications", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_application.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_cross_application.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    shortest = load_shortest()
    cross = load_cross_application()
    shortest_end_to_end(shortest)
    shortest_breakdown(shortest)
    cross_application(cross)
    print("Generated shortest-path and cross-application figures in", OUT)


if __name__ == "__main__":
    main()
