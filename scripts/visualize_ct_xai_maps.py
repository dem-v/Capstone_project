"""Backward-compatible shim for `scripts/visualize_ct_xai_maps.py`.

The implementation now lives in `explainai_thesis.cli.commands.visualize_ct_xai_maps` and is the
single source of truth. This file preserves the historical
`python scripts/visualize_ct_xai_maps.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis visualize-ct-xai-maps`.
"""
from explainai_thesis.cli.commands.visualize_ct_xai_maps import main

if __name__ == "__main__":
    main()
