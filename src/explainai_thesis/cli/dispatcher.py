"""Top-level dispatcher for the bundled ``explainai-thesis`` console command.

Usage::

    explainai-thesis                 # list all commands
    explainai-thesis --list          # list all commands
    explainai-thesis <command> ...   # run a command with its native flags
    explainai-thesis <command> --help

A command is executed by running its underlying
``explainai_thesis/cli/commands/<x>.py`` module file with ``run_name="__main__"``
so its own ``argparse``/``main()`` runs unchanged. This guarantees byte-for-byte
identical behaviour and output to the legacy ``python scripts/<x>.py ...``
invocation (whose shim imports the same module); the dispatcher only provides a
single, discoverable, installable entry point over the migrated command set.
"""
from __future__ import annotations

import runpy
import sys
from typing import Sequence

from ._registry import Command, discover_commands


def _format_listing(commands: dict[str, Command]) -> str:
    if not commands:
        return (
            "No commands found. Set EXPLAINAI_COMMANDS_DIR to the "
            "'explainai_thesis/cli/commands' directory, or run from a source "
            "checkout."
        )
    width = max(len(name) for name in commands)
    lines = ["Available commands:", ""]
    for name, cmd in commands.items():
        lines.append(f"  {name.ljust(width)}  {cmd.help}")
    lines.append("")
    lines.append(
        "Run 'explainai-thesis <command> --help' for command-specific flags."
    )
    return "\n".join(lines)


def _run_command(command: Command, argv: Sequence[str]) -> int:
    """Execute a command's module file as if invoked directly by path."""
    saved_argv = sys.argv
    sys.argv = [str(command.path), *argv]
    try:
        runpy.run_path(str(command.path), run_name="__main__")
        return 0
    except SystemExit as exc:  # scripts may call sys.exit / argparse error
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = saved_argv


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = discover_commands()

    if not argv or argv[0] in ("-h", "--help", "--list", "list"):
        print(_format_listing(commands))
        return 0

    name = argv[0]
    rest = argv[1:]
    command = commands.get(name)
    if command is None:
        print(f"Unknown command: {name}\n", file=sys.stderr)
        print(_format_listing(commands), file=sys.stderr)
        return 2

    return _run_command(command, rest)


if __name__ == "__main__":
    raise SystemExit(main())
