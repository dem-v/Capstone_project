"""Backward-compatible shim for `scripts/check_environment.py`.

The implementation now lives in `explainai_thesis.cli.commands.check_environment` and is the
single source of truth. This file preserves the historical
`python scripts/check_environment.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis check-environment`.
"""
from explainai_thesis.cli.commands.check_environment import main

if __name__ == "__main__":
    main()
