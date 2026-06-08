"""Backward-compatible shim for `scripts/visualize_cxr_classifier_outcome_thresholds.py`.

The implementation now lives in `explainai_thesis.cli.commands.visualize_cxr_classifier_outcome_thresholds` and is the
single source of truth. This file preserves the historical
`python scripts/visualize_cxr_classifier_outcome_thresholds.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis visualize-cxr-classifier-outcome-thresholds`.
"""
from explainai_thesis.cli.commands.visualize_cxr_classifier_outcome_thresholds import main

if __name__ == "__main__":
    main()
