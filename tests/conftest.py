"""Shared pytest fixtures and helpers for the explainai_thesis test suite.

Conventions
-----------
- All tests assume the package is installed editably (`pip install -e .`).
- CPU is the default device; CUDA tests must be marked ``@pytest.mark.cuda``.
- Slow tests (>5s) must be marked ``@pytest.mark.slow``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _deterministic_torch_seed() -> None:
    """Pin torch RNG state per test for reproducible XAI numerics."""
    torch.manual_seed(0)
