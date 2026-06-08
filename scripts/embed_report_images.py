"""Backward-compatible shim for `scripts/embed_report_images.py`.

The implementation now lives in `explainai_thesis.cli.commands.embed_report_images` and is the
single source of truth. This file preserves the historical
`python scripts/embed_report_images.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis embed-report-images`.
"""
from explainai_thesis.cli.commands.embed_report_images import main

if __name__ == "__main__":
    main()
