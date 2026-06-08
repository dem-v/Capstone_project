"""Backward-compatible shim for `scripts/probe_taheera_vit_eval.py`.

The implementation now lives in `explainai_thesis.cli.commands.probe_taheera_vit_eval` and is the
single source of truth. This file preserves the historical
`python scripts/probe_taheera_vit_eval.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis probe-taheera-vit-eval`.
"""
from explainai_thesis.cli.commands.probe_taheera_vit_eval import main

if __name__ == "__main__":
    main()
