"""Backward-compatible shim for `scripts/tabulate_improvement_paired.py`.

The implementation now lives in `explainai_thesis.cli.commands.tabulate_improvement_paired` and is the
single source of truth. This file preserves the historical
`python scripts/tabulate_improvement_paired.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis tabulate-improvement-paired`.
"""
from explainai_thesis.cli.commands.tabulate_improvement_paired import main

if __name__ == "__main__":
    main()
