"""Bundled command implementations for the ``explainai-thesis`` CLI.

Each module here is a migrated thesis-pipeline / utility script. It is the
single source of truth for that command's logic and is importable as
``explainai_thesis.cli.commands.<module>``. Every module exposes a top-level
``main()`` (and most a ``parse_args()``) and keeps its original ``argparse``
flags/defaults and I/O **unchanged**, so produced output (CSV schemas, PNG
layout, folder names, numeric results) is identical to the historical
``scripts/<x>.py`` invocation.

The thin ``scripts/<x>.py`` files remain as backward-compatible shims that import
``main`` from the matching module here, preserving the documented
``python scripts/<x>.py ...`` paths.
"""
