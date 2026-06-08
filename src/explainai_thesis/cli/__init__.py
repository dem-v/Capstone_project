from __future__ import annotations

from .dispatcher import main
from ._registry import Command, commands_dir, discover_commands, scripts_dir

__all__ = ["main", "Command", "commands_dir", "discover_commands", "scripts_dir"]
