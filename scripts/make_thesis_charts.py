"""Backward-compatible shim for `scripts/make_thesis_charts.py`.

The implementation now lives in `explainai_thesis.cli.commands.make_thesis_charts` and is the
single source of truth. This file preserves the historical
`python scripts/make_thesis_charts.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis make-thesis-charts`.
"""
from explainai_thesis.cli.commands.make_thesis_charts import main

if __name__ == "__main__":
    main()
