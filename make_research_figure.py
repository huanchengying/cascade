#!/usr/bin/env python3
"""Scientific figures from complete warm-start audit records."""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def main():
    records = json.loads(Path("submission_warm_20261007.json").read_text())["records"]
    datasets = ("citibike-100k-distance", "citibike-100k-duration")
    methods = ("warm", "warm_closure_weight", "warm_closure_root", "warm_closure_deadline")
    labels = ("Full warm MILP", "Weight-only pool", "Root-priced pool", "Deadline-priced pool")
    colors = ("#63666a", "#b37a24", "#207b71", "#a63d53")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for axis, dataset in zip(axes, datasets):
        baseline = statistics.median(record["baseline_cost"] for record in records if record["dataset"] == dataset)
        costs = [statistics.median(record["cost"] for record in records if record["dataset"] == dataset and record["method"] == method) / baseline for method in methods]
        bars = axis.bar(np.arange(len(methods)), costs, color=colors, width=0.64)
        axis.set_ylim(0, 1.13)
        axis.set_xticks(np.arange(len(methods)), ("Full", "Weight", "Root", "Deadline"))
        axis.set_title("Distance cost (meters)" if "distance" in dataset else "Recorded duration cost (seconds)", fontsize=10)
        axis.set_ylabel("Cost / independent-path cost")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.set_axisbelow(True)
        for bar, cost in zip(bars, costs):
            axis.text(bar.get_x() + bar.get_width() / 2, cost + 0.025, f"{cost:.3f}", ha="center", fontsize=8)
    figure.legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(colors, labels)], loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    destination = Path("work/research_2026_10_07/priced_ablation")
    figure.savefig(destination.with_suffix(".pdf"))
    figure.savefig(destination.with_suffix(".png"), dpi=180)
    print("wrote", destination)


if __name__ == "__main__":
    main()
