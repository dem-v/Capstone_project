#!/usr/bin/env python3
"""Produce a self-contained `_final.md` from a weekly report draft.

Mirrors the existing `week_N_report_final.md` convention: every local
image reference `![alt](relative/path.png)` is inlined as a
`data:image/png;base64,...` URI so the final markdown (and its PDF
export) renders without access to the repository. Remote URLs and
already-embedded data URIs are left untouched. Errors loudly if any
referenced local image is missing, so a final report never silently
ships a broken image.
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

IMG_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}


def embed(report_path: Path) -> Path:
    base_dir = report_path.parent
    text = report_path.read_text(encoding="utf-8")
    missing: list[str] = []
    embedded = 0

    def repl(match: re.Match) -> str:
        nonlocal embedded
        prefix, src, suffix = match.group(1), match.group(2).strip(), match.group(3)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        img_path = (base_dir / src).resolve()
        if not img_path.is_file():
            missing.append(src)
            return match.group(0)
        mime = MIME.get(img_path.suffix.lower(), "image/png")
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        embedded += 1
        return f"{prefix}data:{mime};base64,{b64}{suffix}"

    out_text = IMG_RE.sub(repl, text)
    if missing:
        print(f"ERROR: {len(missing)} referenced image(s) not found:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        raise SystemExit(1)

    out_path = report_path.with_name(report_path.stem + "_final.md")
    out_path.write_text(out_text, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"{report_path.name} -> {out_path.name}: embedded {embedded} image(s), {size_mb:.2f} MB")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Draft report .md files")
    args = parser.parse_args()
    for r in args.reports:
        embed(Path(r))


if __name__ == "__main__":
    main()
