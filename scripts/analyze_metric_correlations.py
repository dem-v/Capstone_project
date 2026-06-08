"""Backward-compatible shim for `scripts/analyze_metric_correlations.py`.

The implementation now lives in `explainai_thesis.cli.commands.analyze_metric_correlations` and is the
single source of truth. This file preserves the historical
`python scripts/analyze_metric_correlations.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis analyze-metric-correlations`.
"""
from explainai_thesis.cli.commands.analyze_metric_correlations import main

if __name__ == "__main__":
    main()
