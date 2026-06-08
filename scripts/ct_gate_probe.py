"""Backward-compatible shim for `scripts/ct_gate_probe.py`.

The implementation now lives in `explainai_thesis.cli.commands.ct_gate_probe` and is the
single source of truth. This file preserves the historical
`python scripts/ct_gate_probe.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis ct-gate-probe`.
"""
from explainai_thesis.cli.commands.ct_gate_probe import main

if __name__ == "__main__":
    main()
