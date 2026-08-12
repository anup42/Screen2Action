#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! "${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  INSTALL_COMMAND="${SCREEN2ACTION_TORCH_INSTALL_CMD:-}"
  case "${INSTALL_COMMAND}" in
    *"python -m pip install"*torch*) sh -c "${INSTALL_COMMAND}" ;;
    *)
      echo "Install CUDA PyTorch with the official selector first, or set" >&2
      echo "SCREEN2ACTION_TORCH_INSTALL_CMD to an explicit python -m pip install ... torch command." >&2
      exit 2
      ;;
  esac
fi

"${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available(), 'CUDA PyTorch is unavailable'"
"${PYTHON_BIN}" -m pip install -e ".[perception,data,train,export,dev]"
"${PYTHON_BIN}" -m screen2action doctor --device cuda --json
