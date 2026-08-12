#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install -e ".[cpu,dev,export]"
"${PYTHON_BIN}" -m screen2action doctor --device cpu --json
"${PYTHON_BIN}" -m screen2action.tools.verify all
