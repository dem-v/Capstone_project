from __future__ import annotations

import argparse

import torch


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


# Shared argparse helpers. Per-script default values are overridable so
# existing CLI defaults (AGENT.md hard constraint) remain frozen; only
# the help text and flag names are canonical here.


def add_manifest_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str = "data/cxr_pneumothorax_manifest.csv",
) -> None:
    parser.add_argument(
        "--manifest",
        default=default,
        help="Input manifest CSV.",
    )


def add_split_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str = "test",
    choices: tuple[str, ...] = ("train", "test", "any"),
    help: str | None = "Manifest split to sample.",
) -> None:
    parser.add_argument(
        "--split",
        default=default,
        choices=list(choices),
        help=help,
    )


def add_output_dir_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str,
    help: str = "Output directory.",
) -> None:
    parser.add_argument("--output-dir", default=default, help=help)


def add_device_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str = "auto",
    help: str | None = "Compute device. `auto` prefers CUDA when available.",
) -> None:
    parser.add_argument(
        "--device",
        default=default,
        choices=["auto", "cpu", "cuda"],
        help=help,
    )


def add_seed_arg(
    parser: argparse.ArgumentParser,
    *,
    default: int = 20260515,
) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=default,
        help="Random seed for reproducible sampling and PRNG-touching XAI methods.",
    )
