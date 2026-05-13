#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
from pathlib import Path


PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "PIL",
    "cv2",
    "pydicom",
    "nibabel",
    "captum",
    "shap",
    "pytorch_grad_cam",
    "torchxrayvision",
]


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.cwd() / ".kaggle"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["KAGGLE_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    failures = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            print(f"OK {package}: {version}")
        except Exception as exc:
            failures.append((package, str(exc)))
            print(f"FAIL {package}: {exc}")

    if failures:
        raise SystemExit(1)

    kaggle_json = Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json"
    if kaggle_json.exists():
        print(f"OK Kaggle credentials found at {kaggle_json}")
    else:
        print(f"NOTE Kaggle credentials not found at {kaggle_json}")


if __name__ == "__main__":
    main()
