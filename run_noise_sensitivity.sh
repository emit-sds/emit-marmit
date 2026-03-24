#!/bin/bash

# Run MARMIT noise sensitivity analysis (simulation-based retrieval test)
# Usage: bash run_noise_sensitivity.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "MARMIT Noise Sensitivity Analysis"
echo "============================================================"
echo ""

python "$SCRIPT_DIR/scripts/simulation/run_noise_sensitivity_analysis.py"
