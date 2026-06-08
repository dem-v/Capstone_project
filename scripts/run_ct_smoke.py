"""Backward-compatible shim for `scripts/run_ct_smoke.py`.

The implementation now lives in `explainai_thesis.cli.commands.run_ct_smoke` and is the
single source of truth. This file preserves the historical
`python scripts/run_ct_smoke.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis run-ct-smoke`.
"""
from explainai_thesis.cli.commands.run_ct_smoke import main

if __name__ == "__main__":
    main()
