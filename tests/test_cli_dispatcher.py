"""Tests for the bundled ``explainai-thesis`` CLI dispatcher.

These guard the discovery + dispatch layer only. They do NOT run any thesis
pipeline (no GPU, no experiment output); the dispatcher's contract is that it
runs each ``scripts/<x>.py`` unchanged, so the per-script behaviour is covered
by the existing golden/smoke tests.
"""
from __future__ import annotations

from explainai_thesis import cli
from explainai_thesis.cli import _registry


def test_discovers_known_commands():
    commands = _registry.discover_commands()
    # A representative set of thesis-pipeline commands must be discoverable.
    for name in (
        "make-thesis-charts",
        "run-cxr-torchxray-smoke",
        "calibrate-cxr-xai-thresholds",
        "visualize-cxr-classifier-outcome-thresholds",
        "run-ct-smoke",
    ):
        assert name in commands, name
        assert commands[name].path.is_file()


def test_command_name_is_kebab_case_without_leading_underscore():
    assert _registry._command_name("run_cxr_torchxray_smoke") == "run-cxr-torchxray-smoke"
    assert _registry._command_name("_fix_progress_encoding") == "fix-progress-encoding"


def test_list_returns_zero(capsys):
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "Available commands:" in out
    assert "make-thesis-charts" in out


def test_no_args_lists_commands(capsys):
    assert cli.main([]) == 0
    assert "Available commands:" in capsys.readouterr().out


def test_unknown_command_returns_two(capsys):
    assert cli.main(["definitely-not-a-command"]) == 2
    assert "Unknown command" in capsys.readouterr().err
