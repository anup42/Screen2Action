"""Secret-safe environment diagnostics for CPU and CUDA operators."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from screen2action import CONFIG_SCHEMA_VERSION, __version__

FEATURE_MODULES = {
    "core": ("yaml",),
    "cpu": ("torch",),
    "perception": (
        "huggingface_hub",
        "ultralytics",
        "doctr",
        "torchvision",
        "timm",
        "transformers",
    ),
    "data": ("datasets", "numpy", "PIL", "pyarrow", "webdataset"),
    "train": ("torch", "tensorboard"),
    "android_control": ("tensorflow",),
    "export": ("onnx", "onnxruntime"),
    "dev": ("pytest", "ruff", "mypy"),
}
MODULE_DISTRIBUTIONS = {
    "yaml": "PyYAML",
    "huggingface_hub": "huggingface-hub",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "doctr": "python-doctr",
    "torchvision": "torchvision",
    "timm": "timm",
    "transformers": "transformers",
    "datasets": "datasets",
    "numpy": "numpy",
    "PIL": "Pillow",
    "pyarrow": "pyarrow",
    "webdataset": "webdataset",
    "tensorboard": "tensorboard",
    "tensorflow": "tensorflow",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "pytest": "pytest",
    "ruff": "ruff",
    "mypy": "mypy",
}
ROOT_VARIABLES = {
    "data": "SCREEN2ACTION_DATA_ROOT",
    "cache": "SCREEN2ACTION_CACHE_ROOT",
    "runs": "SCREEN2ACTION_RUN_ROOT",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nearest_existing(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _root_state(
    repository_root: Path,
    environment: Mapping[str, str],
) -> tuple[dict[str, dict[str, object]], dict[str, Path]]:
    states: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    for name, variable in ROOT_VARIABLES.items():
        configured = variable in environment
        path = (
            Path(environment[variable]).expanduser().resolve()
            if configured
            else (repository_root / name).resolve()
        )
        probe = _nearest_existing(path)
        disk = shutil.disk_usage(probe)
        paths[name] = path
        states[name] = {
            "environment_variable": variable,
            "display_path": f"${{{variable}}}",
            "configured": configured,
            "exists": path.exists(),
            "writable": os.access(probe, os.W_OK),
            "free_bytes": disk.free,
        }
    return states, paths


def _memory_state() -> dict[str, int | None]:
    try:
        psutil = importlib.import_module("psutil")
        memory = psutil.virtual_memory()
        return {"total_bytes": int(memory.total), "available_bytes": int(memory.available)}
    except (ImportError, AttributeError):
        pass
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        windll = getattr(ctypes, "windll", None)
        if windll is not None and windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {
                "total_bytes": int(status.total_physical),
                "available_bytes": int(status.available_physical),
            }
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
            available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
            return {"total_bytes": total, "available_bytes": available}
        except (OSError, ValueError):
            pass
    return {"total_bytes": None, "available_bytes": None}


def _torch_state(requested_device: str) -> dict[str, object]:
    if not _module_available("torch"):
        return {
            "installed": False,
            "version": None,
            "compiled_cuda": None,
            "cuda_available": False,
            "gpu_count": 0,
            "gpus": [],
            "nccl_available": False,
        }
    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    gpus: list[dict[str, object]] = []
    if requested_device == "cuda" and cuda_available:
        for index in range(int(torch.cuda.device_count())):
            properties = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            gpus.append(
                {
                    "index": index,
                    "name": str(properties.name),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_vram_bytes": int(total_bytes),
                    "free_vram_bytes": int(free_bytes),
                }
            )
    distributed = getattr(torch, "distributed", None)
    nccl_available = bool(
        distributed is not None and distributed.is_available() and distributed.is_nccl_available()
    )
    return {
        "installed": True,
        "version": str(torch.__version__),
        "compiled_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "gpu_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "gpus": gpus,
        "nccl_available": nccl_available,
    }


def _feature_state() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for feature, modules in FEATURE_MODULES.items():
        availability = {module: _module_available(module) for module in modules}
        result[feature] = {"ready": all(availability.values()), "modules": availability}
    return result


def _package_state() -> dict[str, str | None]:
    """Report versions for every optional feature distribution without importing it."""

    distributions = sorted(set(MODULE_DISTRIBUTIONS.values()), key=str.casefold)
    return {distribution: _package_version(distribution) for distribution in distributions}


def _model_state(repository_root: Path, cache_root: Path) -> dict[str, object]:
    registry = repository_root / "configs" / "models" / "registry.yaml"
    lock = repository_root / "configs" / "models" / "lock.json"
    locked_files = 0
    verified_files = 0
    malformed = False
    if lock.is_file():
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
            entries = payload.get("models", [])
            if not isinstance(entries, list):
                raise ValueError("models must be a list")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("model entry must be a mapping")
                for file_entry in entry.get("files", []):
                    locked_files += 1
                    relative = Path(str(file_entry["path"]))
                    expected = str(file_entry["sha256"])
                    candidate = cache_root / relative
                    if candidate.is_file() and _sha256(candidate) == expected:
                        verified_files += 1
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            malformed = True
    return {
        "registry_present": registry.is_file(),
        "lock_present": lock.is_file(),
        "lock_malformed": malformed,
        "locked_file_count": locked_files,
        "verified_file_count": verified_files,
        "weights_ready": lock.is_file() and not malformed and locked_files == verified_files,
    }


def _external_state(data_root: Path, cache_root: Path, run_root: Path) -> dict[str, object]:
    data_manifests = (
        list(data_root.glob("normalized/*/manifests/*.json")) if data_root.exists() else []
    )
    cache_manifests = (
        list(cache_root.glob("perception/**/manifest.json")) if cache_root.exists() else []
    )
    checkpoints = list(run_root.glob("**/*.pt")) if run_root.exists() else []
    return {
        "data": {
            "manifest_count": len(data_manifests),
            "compatible": None if not data_manifests else "requires_manifest_validation",
        },
        "perception_cache": {
            "manifest_count": len(cache_manifests),
            "compatible": None if not cache_manifests else "requires_cache_validation",
        },
        "checkpoints": {
            "count": len(checkpoints),
            "compatible": None if not checkpoints else "requires_checkpoint_validation",
        },
    }


def doctor_report(
    repository_root: str | Path,
    *,
    device: str = "cpu",
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Collect diagnostics without downloading assets or exposing environment values."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    root = Path(repository_root).resolve()
    env = os.environ if environment is None else environment
    roots, paths = _root_state(root, env)
    torch_state = _torch_state(device)
    features = _feature_state()
    models = _model_state(root, paths["cache"])
    external = _external_state(paths["data"], paths["cache"], paths["runs"])
    warnings: list[str] = []
    if not torch_state["installed"]:
        warnings.append("PyTorch is not installed; install the cpu extra or an official GPU build.")
    if device == "cuda" and not torch_state["cuda_available"]:
        warnings.append("CUDA was requested but torch.cuda.is_available() is false.")
    gpu_count = torch_state["gpu_count"]
    if (
        device == "cuda"
        and isinstance(gpu_count, int)
        and gpu_count > 1
        and not torch_state["nccl_available"]
    ):
        warnings.append("Multiple GPUs are visible but the installed PyTorch build lacks NCCL.")
    for name, state in roots.items():
        if not state["configured"]:
            warnings.append(
                f"{state['environment_variable']} is unset; using ignored repository-local {name}."
            )
        if not state["writable"]:
            warnings.append(f"configured {name} root is not writable.")
    if not models["lock_present"]:
        warnings.append("The model registry has not been resolved to an immutable lock file.")
    if models["lock_malformed"]:
        warnings.append("The model lock file is malformed.")
    ok = bool(torch_state["installed"]) and all(bool(state["writable"]) for state in roots.values())
    if device == "cuda":
        ok = ok and bool(torch_state["cuda_available"])
    if not torch_state["installed"]:
        next_command = "python -m pip install -e .[cpu,dev]"
    elif device == "cuda" and not torch_state["cuda_available"]:
        next_command = "Install CUDA PyTorch with the official selector, then rerun doctor."
    elif not models["lock_present"]:
        next_command = "screen2action models resolve-lock --dry-run --json"
    else:
        next_command = f"screen2action train smoke --device {device}"
    return {
        "ok": ok,
        "requested_device": device,
        "screen2action": {
            "version": __version__,
            "config_schema_version": CONFIG_SCHEMA_VERSION,
        },
        "host": {
            "python": platform.python_version(),
            "python_executable_name": Path(sys.executable).name,
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "ram": _memory_state(),
        },
        "packages": _package_state(),
        "torch": torch_state,
        "roots": roots,
        "features": features,
        "models": models,
        "external_state": external,
        "warnings": warnings,
        "recommended_next_command": next_command,
    }
