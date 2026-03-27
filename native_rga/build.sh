#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --user pybind11 setuptools wheel
python3 setup.py build_ext --inplace

# Copy module to project root so default import path works.
cp -f rockchip_rga*.so ../
echo "Built module: ../rockchip_rga*.so"
