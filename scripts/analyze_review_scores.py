"""Backward-compatible shim for `scripts/analyze_review_scores.py`.

The implementation now lives in `explainai_thesis.cli.commands.analyze_review_scores` and is the
single source of truth. This file preserves the historical
`python scripts/analyze_review_scores.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis analyze-review-scores`.
"""
from explainai_thesis.cli.commands.analyze_review_scores import main

if __name__ == "__main__":
    main()
