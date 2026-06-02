"""Head-CT hemorrhage pilot (Phase 5.4).

Gate-first by pre-mortem design: the model/manifest/smoke modules are
Branch-A artifacts that are only authorized after the hour-1
model-and-mask availability check passes (see
`docs/refactor_plan.md` § "Phase 5.4"). This package currently ships
only the locked, network-free I/O primitives (`io.py`): the brain-window
HU transform (WL=40, WW=80) and slice handling, which are valid
regardless of the gate outcome.
"""
