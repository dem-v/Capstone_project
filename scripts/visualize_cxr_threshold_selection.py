"""Backward-compatible shim for `scripts/visualize_cxr_threshold_selection.py`.

The implementation now lives in `explainai_thesis.cli.commands.visualize_cxr_threshold_selection` and is the
single source of truth. This file preserves the historical
`python scripts/visualize_cxr_threshold_selection.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis visualize-cxr-threshold-selection`.
"""
from explainai_thesis.cli.commands.visualize_cxr_threshold_selection import main

if __name__ == "__main__":
    main()
