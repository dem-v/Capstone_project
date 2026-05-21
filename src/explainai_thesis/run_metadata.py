"""Run metadata helpers for experiment output folders."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_short_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def build_run_metadata(args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    """Return a JSON-serializable metadata snapshot for a script run."""

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "args": _json_safe(vars(args)),
        "extra": _json_safe(extra),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torchxrayvision_version": _package_version("torchxrayvision"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        },
        "git": {
            "short_hash": _git_short_hash(),
        },
    }


def write_run_metadata(output_dir: Path, args: argparse.Namespace, **extra: Any) -> Path:
    """Write `run_meta.json` to an experiment output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_meta.json"
    path.write_text(
        json.dumps(build_run_metadata(args, **extra), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path