"""Backward-compatible shim for `scripts/build_ct_manifest.py`.

The implementation now lives in `explainai_thesis.cli.commands.build_ct_manifest` and is the
single source of truth. This file preserves the historical
`python scripts/build_ct_manifest.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis build-ct-manifest`.
"""
from explainai_thesis.cli.commands.build_ct_manifest import main

if __name__ == "__main__":
    main()
