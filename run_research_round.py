#!/usr/bin/env python3
"""Serialize benchmark batches after all measured source files are frozen."""

import os
import subprocess
import sys


def main():
    commands = [
        ["benchmark_submission.py", "--config", "work/research_2026_10_07/bitcoin_config.json", "--output", "submission_bitcoin_20261007_frozen.json"],
        ["benchmark_warm_start.py", "--config", "work/research_2026_10_07/sensitivity/config.json", "--output", "submission_sensitivity_20261007.json"],
        ["benchmark_speedsteiner.py", "--config", "work/research_2026_10_07/exact_samples/config.json", "--output", "submission_speedsteiner_exact_20261007.json"],
        ["benchmark_speedsteiner.py", "--config", "work/research_2026_10_07/sensitivity/config.json", "--output", "submission_speedsteiner_sensitivity_20261007.json"],
    ]
    for command in commands:
        print("START " + " ".join(command), flush=True)
        subprocess.run([sys.executable, *command], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        print("DONE " + command[-1], flush=True)


if __name__ == "__main__":
    main()
