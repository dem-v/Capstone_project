"""Backward-compatible shim for `scripts/run_ct_improvement_experiment.py`.

The implementation now lives in `explainai_thesis.cli.commands.run_ct_improvement_experiment` and is the
single source of truth. This file preserves the historical
`python scripts/run_ct_improvement_experiment.py ...` invocation path; behaviour and output are
unchanged. The bundled equivalent is `explainai-thesis run-ct-improvement-experiment`.
"""
from explainai_thesis.cli.commands.run_ct_improvement_experiment import main

if __name__ == "__main__":
    main()
