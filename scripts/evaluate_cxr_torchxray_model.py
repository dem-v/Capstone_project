"""Backward-compatible shim for `scripts/evaluate_cxr_torchxray_model.py`.

The implementation now lives in `explainai_thesis.cli.commands.evaluate_cxr_torchxray_model` and is the
single source of truth. This file preserves the historical
`python scripts/evaluate_cxr_torchxray_model.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis evaluate-cxr-torchxray-model`.
"""
from explainai_thesis.cli.commands.evaluate_cxr_torchxray_model import main

if __name__ == "__main__":
    main()
