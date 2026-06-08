"""Backward-compatible shim for `scripts/diagnose_cxr_torchxray_baselines.py`.

The implementation now lives in `explainai_thesis.cli.commands.diagnose_cxr_torchxray_baselines` and is the
single source of truth. This file preserves the historical
`python scripts/diagnose_cxr_torchxray_baselines.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis diagnose-cxr-torchxray-baselines`.
"""
from explainai_thesis.cli.commands.diagnose_cxr_torchxray_baselines import main

if __name__ == "__main__":
    main()
