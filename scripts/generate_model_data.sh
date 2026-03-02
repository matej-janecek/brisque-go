#!/bin/bash
# Download OpenCV BRISQUE model files and convert to Go source code.
#
# Usage:
#   cd scripts/
#   bash generate_model_data.sh
#
# This will create ../model_default.go with all embedded model data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_URL="https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml"
RANGE_URL="https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml"

echo "Downloading brisque_model_live.yml..."
curl -fsSL -o brisque_model_live.yml "$MODEL_URL"

echo "Downloading brisque_range_live.yml..."
curl -fsSL -o brisque_range_live.yml "$RANGE_URL"

echo "Converting to Go source code..."
python3 convert_model.py brisque_model_live.yml brisque_range_live.yml > ../model_default.go

echo "Running gofmt..."
gofmt -w ../model_default.go

echo "Done! Created model_default.go"
echo "SVs and feature count:"
grep -c "// SV " ../model_default.go || true
echo "lines in file:"
wc -l < ../model_default.go
