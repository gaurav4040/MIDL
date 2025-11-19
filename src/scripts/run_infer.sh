#!/usr/bin/env bash
if [ -z "$1" ]; then
  echo "Usage: bash scripts/run_infer.sh /path/to/volume.nii.gz"
  exit 1
fi
python -m src.inference.infer --vol "$1" --cfg configs/config.yaml
