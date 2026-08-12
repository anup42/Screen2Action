"""Offline verification entry point used by Makefile and Windows workflows."""

from __future__ import annotations

import argparse
import ast
import compileall
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIRS = (ROOT / "src", ROOT / "tests")


def _python_files() -> list[Path]:
    return [path for directory in PYTHON_DIRS for path in directory.rglob("*.py")]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    current_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = source_path + (os.pathsep + current_path if current_path else "")
    return environment


def format_check() -> int:
    """Check syntax and simple formatting invariants without network access."""

    files = _python_files()
    failures: list[str] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as error:
            failures.append(f"{path}: {error}")
        for line_number, line in enumerate(source.splitlines(), start=1):
            if line.rstrip() != line:
                failures.append(f"{path}:{line_number}: trailing whitespace")
    if not compileall.compile_dir(str(ROOT / "src"), quiet=1):
        failures.append("compileall failed for src")
    ruff = shutil.which("ruff")
    if ruff:
        result = subprocess.run(
            [ruff, "format", "--check", "src", "tests"],
            cwd=ROOT,
            env=_environment(),
            check=False,
        )
        if result.returncode:
            return result.returncode
    if failures:
        print("\n".join(failures))
        return 1
    print(f"format-check passed ({len(files)} Python files)")
    return 0


def lint() -> int:
    """Run Ruff when installed and always enforce CPU-only source policy."""

    violations: list[str] = []
    for path in _python_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        if ".cuda(" in source:
            violations.append(f"{path}: direct .cuda() is forbidden")
        if "import torch.cuda" in source or "from torch import cuda" in source:
            violations.append(f"{path}: unconditional CUDA import is forbidden")
    if violations:
        print("\n".join(violations))
        return 1
    ruff = shutil.which("ruff")
    if ruff:
        return subprocess.run(
            [ruff, "check", "src", "tests"],
            cwd=ROOT,
            env=_environment(),
            check=False,
        ).returncode
    print("lint passed (Ruff not installed; local CPU-policy checks ran)")
    return 0


def typecheck() -> int:
    """Run Mypy when installed; package syntax is checked in all environments."""

    mypy = shutil.which("mypy")
    if not mypy:
        print("typecheck skipped: Mypy is not installed")
        return 0
    return subprocess.run(
        [mypy, "src"],
        cwd=ROOT,
        env=_environment(),
        check=False,
    ).returncode


def tests() -> int:
    """Run the offline test suite."""

    return subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=ROOT,
        env=_environment(),
        check=False,
    ).returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=("format", "lint", "typecheck", "test", "all"),
        help="verification target",
    )
    args = parser.parse_args(argv)
    checks = {
        "format": format_check,
        "lint": lint,
        "typecheck": typecheck,
        "test": tests,
    }
    if args.target == "all":
        for check in (format_check, lint, typecheck, tests):
            if check() != 0:
                return 1
        return 0
    return checks[args.target]()


if __name__ == "__main__":
    raise SystemExit(main())
