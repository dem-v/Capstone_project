"""Backward-compatible shim for `scripts/calibrate_cxr_xai_thresholds.py`.

The implementation now lives in `explainai_thesis.cli.commands.calibrate_cxr_xai_thresholds` and is the
single source of truth. This file preserves the historical
`python scripts/calibrate_cxr_xai_thresholds.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis calibrate-cxr-xai-thresholds`.
"""
from explainai_thesis.cli.commands.calibrate_cxr_xai_thresholds import main

if __name__ == "__main__":
    main()
