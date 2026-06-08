"""Backward-compatible shim for `scripts/probe_taheera_vit.py`.

The implementation now lives in `explainai_thesis.cli.commands.probe_taheera_vit` and is the
single source of truth. This file preserves the historical
`python scripts/probe_taheera_vit.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis probe-taheera-vit`.
"""
from explainai_thesis.cli.commands.probe_taheera_vit import main

if __name__ == "__main__":
    main()
