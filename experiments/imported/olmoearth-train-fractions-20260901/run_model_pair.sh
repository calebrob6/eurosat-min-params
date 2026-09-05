#!/usr/bin/env bash
set -euo pipefail

model=$1
batch_size=$2
root=outputs/olmoearth-train-fractions-20260901
runner=outputs/eurosat-train-fractions-20260901/run_fraction_sweep.py

mkdir -p "$root/normal/logs" "$root/spatial/logs"

conda run --no-capture-output -n torchgeo-bench python "$runner" \
  --dataset eurosat \
  --fractions 0.01,0.02,0.05,0.10,0.20,0.50,1.0 \
  --model "$model" \
  --output "$root/normal/$model.csv" \
  --batch-size "$batch_size" \
  --device cuda:0 \
  --skip-knn \
  > "$root/normal/logs/$model.log" 2>&1

conda run --no-capture-output -n torchgeo-bench python "$runner" \
  --dataset eurosat-spatial \
  --fractions 0.01,0.02,0.05,0.10,0.20,0.50,1.0 \
  --model "$model" \
  --output "$root/spatial/$model.csv" \
  --batch-size "$batch_size" \
  --device cuda:0 \
  --skip-knn \
  > "$root/spatial/logs/$model.log" 2>&1
