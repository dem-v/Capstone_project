"""Backward-compatible shim for `scripts/run_smoke_test.py`.

The implementation now lives in `explainai_thesis.cli.commands.run_smoke_test` and is the
single source of truth. This file preserves the historical
`python scripts/run_smoke_test.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis run-smoke-test`.
"""
from explainai_thesis.cli.commands.run_smoke_test import main

if __name__ == "__main__":
    main()
