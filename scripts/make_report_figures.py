"""Backward-compatible shim for `scripts/make_report_figures.py`.

The implementation now lives in `explainai_thesis.cli.commands.make_report_figures` and is the
single source of truth. This file preserves the historical
`python scripts/make_report_figures.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis make-report-figures`.
"""
from explainai_thesis.cli.commands.make_report_figures import main

if __name__ == "__main__":
    main()
