"""Backward-compatible shim for `scripts/select_cxr_review_candidates.py`.

The implementation now lives in `explainai_thesis.cli.commands.select_cxr_review_candidates` and is the
single source of truth. This file preserves the historical
`python scripts/select_cxr_review_candidates.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis select-cxr-review-candidates`.
"""
from explainai_thesis.cli.commands.select_cxr_review_candidates import main

if __name__ == "__main__":
    main()
