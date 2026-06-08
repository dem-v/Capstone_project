"""Backward-compatible shim for `scripts/plot_ct_improvement.py`.

The implementation now lives in `explainai_thesis.cli.commands.plot_ct_improvement` and is the
single source of truth. This file preserves the historical
`python scripts/plot_ct_improvement.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis plot-ct-improvement`.
"""
from explainai_thesis.cli.commands.plot_ct_improvement import main

if __name__ == "__main__":
    main()
