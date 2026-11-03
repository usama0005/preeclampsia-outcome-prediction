#!/usr/bin/env bash
# Run the full pipeline. Usage:
#   DATA_PATH=data/my_data.xlsx bash run_all.sh          # full run
#   SMOKE=1 DATA_PATH=data/my_data.xlsx bash run_all.sh  # fast test run (few imputations/bootstraps)
set -e
for s in 01_descriptives 02_model 03_validation 04_sensitivity 05_nomogram; do
  echo; echo "=== $s ==="; python "$s.py"
done
