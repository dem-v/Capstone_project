"""Backward-compatible shim for `scripts/build_review_workbook.py`.

The implementation now lives in `explainai_thesis.cli.commands.build_review_workbook` and is the
single source of truth. This file preserves the historical
`python scripts/build_review_workbook.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis build-review-workbook`.
"""
from explainai_thesis.cli.commands.build_review_workbook import main

if __name__ == "__main__":
    main()
