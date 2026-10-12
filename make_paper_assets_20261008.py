#!/usr/bin/env python3
"""Generate the October 8 paper tables and matched-budget figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_LABELS = {
    "path_union": "Independent paths",
    "star": "Conservative star",
    "cascade": "\\cascade",
    "huang2": "Huang level 2 reference",
    "huang3": "Huang level 3 reference",
    "speedsteiner": "SpeedSteiner author code",
    "ikuta": "Ikuta edge/prefix ILP",
    "closure_root": "LP-guided restricted ILP",
    "cascade_warm": "Full Ikuta",
    "cascade_warm_root": "Full Ikuta + root cuts",
    "cascade_closure_weight": "Weight-only pool",
    "cascade_closure_root": "LP-guided pool",
}

DATA_LABELS = {
    "citibike-100k-distance": "Citi distance",
    "citibike-100k-duration": "Citi duration",
    "nyc-tlc-100k-distance": "TLC distance",
    "nyc-tlc-100k-duration": "TLC duration",
    "nyc-tlc-100k-total-amount": "TLC total amount",
    "ALL": "All pairs",
}


def _number(value, precision=3):
    return "--" if value is None else f"{value:.{precision}f}"


def _exact_table(payload: dict) -> str:
    rows = []
    for summary in payload["method_summaries"]:
        label = METHOD_LABELS.get(summary["method"], summary["method"])
        rows.append(
            f"{label} & {summary['available_samples']}/{summary['samples']} & "
            f"{summary['mean_cost_ratio']:.5f} & "
            f"{summary['geometric_mean_cost_ratio']:.5f} & "
            f"{summary['worst_cost_ratio']:.5f} & "
            f"{summary['optimal_hits']}/{summary['available_samples']} & "
            f"{summary['median_wall_s']:.4f} s\\\\"
        )
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Quality on 24 frozen real-data samples. Repeats are collapsed before statistics. Quality summaries use available samples; ``Avail.'' retains guarded samples in the coverage denominator.}",
            "\\label{tab:exact}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Method & Avail. & Mean/OPT & Geomean/OPT & Worst/OPT & Optimal & Median time\\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )


def _cascade_table(payload: dict) -> str:
    summaries = payload["summaries"]
    rows = []
    for summary in summaries:
        if summary["dataset"] == "ALL":
            rows.append("\\midrule")
        rows.append(
            f"{DATA_LABELS[summary['dataset']]} & "
            f"{summary['wins']}/{summary['ties']}/{summary['losses']} & "
            f"{summary['verified_converged_baselines']}/{summary['paired_units']} & "
            f"{summary['median_relative_saving_pct']:.3f}\\% & "
            f"{summary['median_wall_ratio']:.3f}$\\times$\\\\"
        )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Four capped \\cascade{} rounds after up to 32 conservative-star rounds on 100K edges. Conv. counts baselines with a recorded no-improvement round.}",
            "\\label{tab:postconvergence}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Weight domain & W/T/L & Conv. & Gain & Time ratio\\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def _budget_table(payload: dict) -> str:
    rows = []
    for summary in payload["summaries"]:
        label = METHOD_LABELS.get(summary["method"], summary["method"])
        rows.append(
            f"{summary['budget_s']:g} & {label} & "
            f"{summary['available_units']}/{summary['units']} & "
            f"{summary['improved_units']}/{summary['units']} & "
            f"{_number(summary['mean_improvement_from_initial_pct'])}\\% & "
            f"{_number(summary['median_global_gap_pct'])}\\% & "
            f"{_number(summary['median_wall_s'], 2)} s & "
            f"{_number(summary['median_peak_rss_mib'], 0)} & "
            f"{summary['over_budget_runs']}\\\\"
        )
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Matched-incumbent results over twenty 100K-edge dataset-root units. Budgets are nominal; actual wall times and overruns are retained. Reduction is relative to the common \\cascade{} incumbent. Gaps use only full-graph bounds.}",
            "\\label{tab:budget}",
            "\\small",
            "\\begin{tabular}{rlrrrrrrr}",
            "\\toprule",
            "Budget & Method & Avail. & Improved & Mean reduction & Global gap & Median wall & Peak RSS (MiB) & Overruns\\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )


def write_figure(payload: dict, pdf_path: Path, png_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = [
        "cascade_warm",
        "cascade_warm_root",
        "cascade_closure_weight",
        "cascade_closure_root",
    ]
    colors = {
        "cascade_warm": "#1f77b4",
        "cascade_warm_root": "#d62728",
        "cascade_closure_weight": "#7f7f7f",
        "cascade_closure_root": "#2ca02c",
    }
    markers = {
        "cascade_warm": "o",
        "cascade_warm_root": "s",
        "cascade_closure_weight": "^",
        "cascade_closure_root": "D",
    }
    summaries = payload["summaries"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.25))
    for method in methods:
        selected = sorted(
            (row for row in summaries if row["method"] == method),
            key=lambda row: row["budget_s"],
        )
        budgets = [row["budget_s"] for row in selected]
        quality = [100.0 * (1.0 - row["geometric_mean_cost_to_initial"]) for row in selected]
        gaps = [row["median_global_gap_pct"] for row in selected]
        label = METHOD_LABELS[method].replace(" + ", "+")
        axes[0].plot(
            budgets,
            quality,
            color=colors[method],
            marker=markers[method],
            linewidth=1.8,
            markersize=5,
            label=label,
        )
        axes[1].plot(
            budgets,
            gaps,
            color=colors[method],
            marker=markers[method],
            linewidth=1.8,
            markersize=5,
            label=label,
        )
    axes[0].set_xlabel("Nominal budget (s)")
    axes[0].set_ylabel("Geomean cost reduction from incumbent (%)")
    axes[1].set_xlabel("Nominal budget (s)")
    axes[1].set_ylabel("Median global gap (%)")
    for axis in axes:
        axis.set_xticks([10, 30, 60])
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.subplots_adjust(top=0.78, bottom=0.2, left=0.085, right=0.985, wspace=0.28)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=180)
    plt.close(figure)


def _convergence_table(payload: dict) -> str:
    rows = []
    for summary in payload["summaries"]:
        if summary["dataset"] == "ALL":
            rows.append("\\midrule")
        rows.append(
            f"{DATA_LABELS[summary['dataset']]} & "
            f"{summary['verified_converged_baselines']}/{summary['paired_units']} & "
            f"{summary['wins']}/{summary['ties']}/{summary['losses']} & "
            f"{summary['median_relative_saving_pct']:.3f}\\% & "
            f"{summary['median_wall_ratio']:.3f}$\\times$\\\\"
        )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Convergence audit: star search continues until no improving neighbor is found, subject to a 512-round safeguard, then applies four capped \\cascade{} rounds.}",
            "\\label{tab:convergence}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Weight domain & Conv. & W/T/L & Gain & Time ratio\\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def _speed_table(payload: dict) -> str:
    rows = []
    for summary in payload["summaries"]:
        if summary["dataset"] == "ALL":
            continue
        guarded = summary["statuses"].get("size-guard", 0)
        rows.append(
            f"{DATA_LABELS[summary['dataset']]} & "
            f"{summary['available_units']}/{summary['units']} & {guarded} & "
            f"{_number(summary['geometric_mean_cost_ratio'])} & "
            f"{_number(summary['median_wall_s'])}\\\\"
        )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{SpeedSteiner author code at 100K edges, nominal ten-second target. Ratios use the longer 32-star/four-Cascade reference; actual wall times differ between methods.}",
            "\\label{tab:speedlarge}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Weight domain & Avail. & Guard & Ratio & Time (s)\\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-summary", type=Path, required=True)
    parser.add_argument("--cascade-summary", type=Path, required=True)
    parser.add_argument("--convergence-summary", type=Path, required=True)
    parser.add_argument("--speed-summary", type=Path, required=True)
    parser.add_argument("--budget-summary", type=Path, required=True)
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("work/research_2026_10_08/paper_tables_20261008.tex"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("work/research_2026_10_08/budget_curve_20261008.pdf"),
    )
    args = parser.parse_args()
    exact = json.loads(args.exact_summary.read_text())
    cascade = json.loads(args.cascade_summary.read_text())
    convergence = json.loads(args.convergence_summary.read_text())
    speed = json.loads(args.speed_summary.read_text())
    budget = json.loads(args.budget_summary.read_text())
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ('exact_table', _exact_table(exact)),
        ('cascade_table', _cascade_table(cascade)),
        ('convergence_table', _convergence_table(convergence)),
        ('speed_table', _speed_table(speed)),
        ('budget_table', _budget_table(budget)),
    ):
        (args.output_tex.parent / f'{name}.tex').write_text(content + '\n')
    args.output_tex.write_text(
        "\n\n".join(
            [
                _exact_table(exact),
                _cascade_table(cascade),
                _convergence_table(convergence),
                _speed_table(speed),
                _budget_table(budget),
            ]
        )
        + "\n"
    )
    write_figure(budget, args.output_figure, args.output_figure.with_suffix(".png"))
    print(f"wrote {args.output_tex} and {args.output_figure}")


if __name__ == "__main__":
    main()
