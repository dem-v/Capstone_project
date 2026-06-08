"""Backward-compatible shim for `scripts/run_improvement_experiment.py`.

The implementation now lives in `explainai_thesis.cli.commands.run_improvement_experiment` and is the
single source of truth. This file preserves the historical
`python scripts/run_improvement_experiment.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis run-improvement-experiment`.
"""
from explainai_thesis.cli.commands.run_improvement_experiment import main

if __name__ == "__main__":
    main()
