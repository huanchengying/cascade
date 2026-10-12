#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=${PYTHON:-work/venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=work/matplotlib
MODE=${1:---check}
if [ "$MODE" != "--check" ] && [ "$MODE" != "--full" ]; then
  printf '%s\n' 'Usage: sh reproduce_20261008.sh [--check|--full]' >&2
  exit 2
fi
if [ "$MODE" = "--check" ]; then
  "$PYTHON" verify_artifact_20261008.py --run-tests
  "$PYTHON" audit_manuscript_consistency_20261009.py --check
  "$PYTHON" audit_citations_20261010.py --check
  "$PYTHON" verify_paper_20261008.py
  exit 0
fi
if [ "$MODE" = "--full" ]; then
  if [ -f manifest.json ]; then
    "$PYTHON" verify_artifact_20261008.py
    mv manifest.json manifest.frozen.json
  fi
  sh build_speedsteiner.sh
  "$PYTHON" prepare_research_workloads.py \
    --input work/data/processed/citibike-202604-distance-100k.json \
    --output work/research_2026_10_08/data/citibike-duration-100k.json \
    --policy recorded-duration
  "$PYTHON" prepare_root_selection_20261008.py
  "$PYTHON" prepare_exact_quality_20261008.py
  "$PYTHON" benchmark_submission.py \
    --config work/research_2026_10_08/exact_samples/config.json \
    --output work/research_2026_10_08/submission_exact_quality_20261008.json
  "$PYTHON" benchmark_speedsteiner.py \
    --config work/research_2026_10_08/exact_samples/config.json \
    --output work/research_2026_10_08/submission_speedsteiner_exact_20261008.json
  "$PYTHON" benchmark_cascade_star.py \
    --config work/research_2026_10_08/cascade_multiroot_config.json \
    --output work/research_2026_10_08/cascade_multiroot_results.json
  "$PYTHON" benchmark_cascade_star.py \
    --config work/research_2026_10_08/cascade_convergence_config.json \
    --output work/research_2026_10_08/cascade_convergence_results.json
  "$PYTHON" benchmark_speedsteiner.py \
    --config work/research_2026_10_08/speedsteiner_100k_config.json \
    --output work/research_2026_10_08/speedsteiner_100k_results.json
  "$PYTHON" benchmark_star_warm.py \
    --config work/research_2026_10_08/budget_curve_config.json \
    --output work/research_2026_10_08/budget_curve_results.json
  "$PYTHON" prepare_extended_20261008.py
  for experiment in microprofile sensitivity additional_exact; do
    "$PYTHON" benchmark_extended_20261008.py \
      --config "work/research_2026_10_08/${experiment}_config.json" \
      --output "work/research_2026_10_08/${experiment}_results.json"
  done
  "$PYTHON" prepare_snap_temporal_20261008.py
  "$PYTHON" benchmark_extended_20261008.py \
    --config work/research_2026_10_08/snap_structural_config.json \
    --output work/research_2026_10_08/snap_structural_results.json
fi
"$PYTHON" verify_artifact_20261008.py --run-tests \
  --output work/research_2026_10_08/verification_20261008.json
"$PYTHON" analyze_exact_quality.py \
  --results work/research_2026_10_08/submission_exact_quality_20261008.json \
  --additional-results work/research_2026_10_08/submission_speedsteiner_exact_20261008.json \
  --optima work/research_2026_10_08/exact_samples/optima.json \
  --output-prefix work/research_2026_10_08/exact_quality_with_speedsteiner
"$PYTHON" analyze_cascade_results.py \
  --results work/research_2026_10_08/cascade_multiroot_results.json \
  --output-prefix work/research_2026_10_08/cascade_multiroot_paired
"$PYTHON" analyze_cascade_results.py \
  --results work/research_2026_10_08/cascade_convergence_results.json \
  --output-prefix work/research_2026_10_08/cascade_convergence_paired
"$PYTHON" analyze_speedsteiner_large.py \
  --speed-results work/research_2026_10_08/speedsteiner_100k_results.json \
  --cascade-results work/research_2026_10_08/cascade_multiroot_results.json \
  --budget-results work/research_2026_10_08/budget_curve_results.json \
  --output-prefix work/research_2026_10_08/speedsteiner_100k_audit
"$PYTHON" analyze_budget_curve.py \
  --results work/research_2026_10_08/budget_curve_results.json \
  --output-prefix work/research_2026_10_08/budget_curve_summary
"$PYTHON" make_paper_assets_20261008.py \
  --exact-summary work/research_2026_10_08/exact_quality_with_speedsteiner.json \
  --cascade-summary work/research_2026_10_08/cascade_multiroot_paired.json \
  --convergence-summary work/research_2026_10_08/cascade_convergence_paired.json \
  --speed-summary work/research_2026_10_08/speedsteiner_100k_audit.json \
  --budget-summary work/research_2026_10_08/budget_curve_summary.json
"$PYTHON" analyze_extended_20261008.py
"$PYTHON" analyze_snap_temporal_20261008.py
"$PYTHON" make_findings_20261008.py
"$PYTHON" make_bibliography_20261008.py
mkdir -p work/paper_build_20261008
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=work/paper_build_20261008 temporal_mst_sigmod2027.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=work/paper_build_20261008 temporal_mst_sigmod2027.tex
mkdir -p output/pdf
cp work/paper_build_20261008/temporal_mst_sigmod2027.pdf \
  output/pdf/temporal_mst_submission_20261008.pdf
cp output/pdf/temporal_mst_submission_20261008.pdf \
  temporal_mst_submission_20261008.pdf
"$PYTHON" verify_paper_20261008.py --build
