"""Command registry for the bundled ``explainai-thesis`` console interface.

The thesis pipeline logic now lives as importable modules under
``explainai_thesis.cli.commands`` (migrated verbatim from the historical flat
``scripts/*.py`` collection). Each module keeps its own ``argparse`` + ``main()``
and a ``__main__`` guard. This module discovers those command modules and
exposes them as named subcommands of a single installable console entry point,
**without changing any behaviour or output**: a subcommand is executed by
running the underlying module file with ``run_name="__main__"`` so the module's
own ``main()`` runs exactly as it does when invoked by path.

The thin ``scripts/<x>.py`` files remain as backward-compatible shims that import
``main`` from the matching command module, so the documented
``python scripts/<x>.py ...`` invocations keep working identically alongside the
``explainai-thesis <command> ...`` form.

The commands directory is resolved relative to this file (the package is
editable-installed from ``src/``), and can be overridden with the
``EXPLAINAI_COMMANDS_DIR`` environment variable (the legacy
``EXPLAINAI_SCRIPTS_DIR`` name is still honoured) for non-standard layouts.
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path


def commands_dir() -> Path:
    """Locate the ``explainai_thesis/cli/commands`` directory.

    Resolution order: ``EXPLAINAI_COMMANDS_DIR`` env var, then the legacy
    ``EXPLAINAI_SCRIPTS_DIR`` env var, then the ``commands`` package directory
    next to this file.
    """
    override = os.environ.get("EXPLAINAI_COMMANDS_DIR") or os.environ.get(
        "EXPLAINAI_SCRIPTS_DIR"
    )
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "commands"


# Backward-compatible alias: the historical public name pointed at the script
# directory; the logic now lives in the commands package.
scripts_dir = commands_dir


def _command_name(stem: str) -> str:
    """Map a module file stem to a kebab-case command name."""
    return stem.lstrip("_").replace("_", "-")


def _short_help(path: Path) -> str:
    """Extract the first non-empty line of the module docstring (no import)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree)
    except (OSError, SyntaxError, ValueError):
        doc = None
    if not doc:
        return ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


@dataclass(frozen=True)
class Command:
    name: str
    path: Path
    help: str


def discover_commands() -> dict[str, Command]:
    """Discover all dispatchable commands from the commands package.

    Returns an ordered (by command name) mapping of command name -> Command.
    Command modules contain a ``__main__`` guard so they can be run by path,
    which every migrated module does.
    """
    base = commands_dir()
    commands: dict[str, Command] = {}
    if not base.is_dir():
        return commands
    for path in sorted(base.glob("*.py")):
        if path.name == "__init__.py":
            continue
        name = _command_name(path.stem)
        commands[name] = Command(name=name, path=path, help=_short_help(path))
    return commands
