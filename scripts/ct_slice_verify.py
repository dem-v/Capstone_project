"""Backward-compatible shim for `scripts/ct_slice_verify.py`.

The implementation now lives in `explainai_thesis.cli.commands.ct_slice_verify` and is the
single source of truth. This file preserves the historical
`python scripts/ct_slice_verify.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis ct-slice-verify`.
"""
from explainai_thesis.cli.commands.ct_slice_verify import main

if __name__ == "__main__":
    main()
