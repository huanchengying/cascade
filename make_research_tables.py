#!/usr/bin/env python3
"""Generate paper tables directly from complete, source-audited experiments."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from summarize_submission import summarize


def load_records(filename):
    path = Path(filename)
    audit = summarize(path)
    if not audit["hashes_match"]:
        raise ValueError(f"{filename}: source/input hash mismatch")
    return json.loads(path.read_text())["records"]


def groups(records):
    result = defaultdict(list)
    for record in records:
        result[record["dataset"], record["root"], record["method"]].append(record)
    return result


def value(records, key):
    return statistics.median(record[key] for record in records if key in record)


def cell(records, key, digits=2):
    if not records or not any(key in record for record in records):
        return "--"
    return f"{value(records, key):,.{digits}f}"


def gap(records):
    values = [100 * max(0.0, (record["cost"] - record["lower_bound"]) / record["cost"]) if record["cost"] else 0.0 for record in records if "cost" in record and "lower_bound" in record]
    return f"{statistics.median(values):.3f}" if values else "--"


def main():
    quality = load_records("submission_exact_quality_20261007.json")
    bitcoin = groups(load_records("submission_bitcoin_20261007_frozen.json"))
    cold = groups(load_records("submission_citi_20261007.json"))
    warm = groups(load_records("submission_warm_20261007.json"))
    sensitivity = groups(load_records("submission_sensitivity_20261007.json"))
    speed_exact = load_records("submission_speedsteiner_exact_20261007.json")
    speed_sensitivity = groups(load_records("submission_speedsteiner_sensitivity_20261007.json"))
    optima = {row["name"]: row["cost"] for row in json.loads(Path("work/research_2026_10_07/exact_samples/optima.json").read_text())}
    lines = []
    lines.extend([r"\begin{table}[t]", r"\centering\small", r"\caption{Twelve exact real induced instances. Ratios use per-instance median cost; three runs per method/instance.}", r"\label{tab:exact}", r"\begin{tabular}{lrrr}\toprule", r"Method & Mean ratio & Max ratio & Hits\\\midrule"])
    names = [("path_union", "Independent paths"), ("huang2", "Huang 2 reference"), ("huang3", "Huang 3 reference"), ("speedsteiner", "SpeedSteiner author"), ("closure_none", "Priced, no cuts"), ("closure_deadline", "Priced, deadlines")]
    all_quality = groups(quality + speed_exact)
    for method, label in names:
        ratios = [value(rows, "cost") / optima[dataset] for (dataset, _, candidate), rows in all_quality.items() if candidate == method and any("cost" in record for record in rows)]
        hits = sum(abs(ratio - 1) <= 1e-7 for ratio in ratios)
        lines.append(f"{label} & {statistics.mean(ratios):.5f} & {max(ratios):.5f} & {hits}/{len(ratios)}" + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    lines.extend([r"\begin{table*}[t]", r"\centering\small", r"\caption{Three-root Bitcoin cost and median end-to-end seconds. All listed optimizer costs equal the full-model optimum; extra pricing is slower.}", r"\label{tab:bitcoin}", r"\begin{tabular}{llrrrr}\toprule", r"Dataset & Root & Paths cost & Full MILP cost / s & Uncut priced cost / s & Deadline priced cost / s\\\midrule"])
    for dataset, root in sorted({(dataset, root) for dataset, root, _ in bitcoin}):
        path_rows = bitcoin[dataset, root, "path_union"]
        full_rows = bitcoin[dataset, root, "ikuta"]
        plain_rows = bitcoin[dataset, root, "closure_none"]
        deadline_rows = bitcoin[dataset, root, "closure_deadline"]
        label = "Alpha" if "alpha" in dataset else "OTC"
        lines.append(f"{label} & {root} & {cell(path_rows,'cost',0)} & {cell(full_rows,'cost',0)} / {cell(full_rows,'wall_time_s')} & {cell(plain_rows,'cost',0)} / {cell(plain_rows,'wall_time_s')} & {cell(deadline_rows,'cost',0)} / {cell(deadline_rows,'wall_time_s')}" + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table*}"])
    lines.extend([r"\begin{table*}[t]", r"\centering\small", r"\caption{Citi Bike 100K at root 6197.08, nominal 30-second allocation. Medians over three runs. Gap is global $(U-L)/U$; times are actual wall seconds and can exceed the allocation.}", r"\label{tab:citi}", r"\begin{tabular}{llrrrr}\toprule", r"Weight & Method & Cost & Gap (\%) & Wall s & Peak MiB\\\midrule"])
    methods = [("path_union", "Independent paths", cold), ("ikuta", "Cold compact MILP", cold), ("ikuta_precut", "Cold + root precuts", cold), ("warm", "Warm full MILP", warm), ("warm_precut", "Warm + root precuts", warm), ("warm_closure_weight", "Warm weight pool", warm), ("warm_closure_root", "Warm root-priced pool", warm), ("warm_closure_deadline", "Warm deadline pool", warm)]
    for dataset in ("citibike-100k-distance", "citibike-100k-duration"):
        for method, label, table in methods:
            rows = table[dataset, "6197.08", method]
            weight = "meters" if "distance" in dataset else "seconds"
            lines.append(f"{weight} & {label} & {cell(rows,'cost',0)} & {gap(rows)} & {cell(rows,'wall_time_s')} & {cell(rows,'peak_rss_mib',0)}" + r"\\")
        if "distance" in dataset:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table*}"])
    lines.extend([r"\begin{table*}[t]", r"\centering\small", r"\caption{Alternative trust penalties and daily mobility windows, nominal 10 seconds. Each cell is median original cost / wall seconds. D1/D2 denote two daily windows; S and M denote seconds and meters.}", r"\label{tab:sensitivity}", r"\begin{tabular}{llrrrr}\toprule", r"Scenario & Root & Warm full & Weight pool & Deadline pool & SpeedSteiner author\\\midrule"])
    labels = {"alpha-trust-positive": "Alpha positive", "alpha-trust-exponential": "Alpha exponential", "otc-trust-positive": "OTC positive", "otc-trust-exponential": "OTC exponential", "citi-day1-geographic-distance": "Citi D1 M", "citi-day1-recorded-duration": "Citi D1 S", "citi-day2-geographic-distance": "Citi D2 M", "citi-day2-recorded-duration": "Citi D2 S"}
    for dataset, root in sorted({(dataset, root) for dataset, root, _ in sensitivity}):
        cells = []
        for method in ("warm", "warm_closure_weight", "warm_closure_deadline", "speedsteiner"):
            table = speed_sensitivity if method == "speedsteiner" else sensitivity
            rows = table[dataset, root, method]
            if any("cost" in record for record in rows):
                cells.append(cell(rows, "cost", 1) + " / " + cell(rows, "wall_time_s"))
            else:
                cells.append("Guard" if all(record.get("status") == "size-guard" for record in rows) else "Timeout")
        lines.append(labels[dataset] + " & " + root + " & " + " & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table*}"])
    destination = Path("work/research_2026_10_07/paper_tables.tex")
    destination.write_text("\n".join(lines) + "\n")
    print("wrote", destination)


if __name__ == "__main__":
    main()
