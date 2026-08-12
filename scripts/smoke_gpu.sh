#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m screen2action doctor --device cuda --json
"${PYTHON_BIN}" -m screen2action train smoke --device cuda --json
