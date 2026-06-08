"""Enable ``python -m explainai_thesis <command> ...`` as an alias for the
bundled ``explainai-thesis`` console entry point."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
