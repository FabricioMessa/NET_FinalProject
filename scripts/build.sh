#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

cmake -S . -B build \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    -DPython_EXECUTABLE="$(which python)"

cmake --build build