"""Backward-compatible shim for `scripts/_fix_progress_encoding.py`.

The implementation now lives in `explainai_thesis.cli.commands._fix_progress_encoding` and is the
single source of truth. This file preserves the historical
`python scripts/_fix_progress_encoding.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis fix-progress-encoding`.
"""
from explainai_thesis.cli.commands._fix_progress_encoding import main

if __name__ == "__main__":
    main()
