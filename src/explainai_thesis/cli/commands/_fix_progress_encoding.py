"""One-shot utility to repair mojibake in docs/progress.md.

The file is UTF-8 but contains sequences that were decoded as cp1252 and
re-encoded as UTF-8 (classic mojibake), plus at least one stray invalid byte.
We:
  1) Replace known mojibake sequences with their intended unicode.
  2) Drop any remaining bytes that cannot be decoded as UTF-8.

Idempotent: running twice produces the same content as running once.
"""
from __future__ import annotations

from pathlib import Path

PATH = Path("docs/progress.md")

# Known mojibake -> intended character.
# These are the cp1252 -> utf-8 double-encodings actually present in the file.
# Mojibake = UTF-8 bytes for a char that were then decoded as cp1252 and
# re-encoded as UTF-8. Sequences below were observed in the actual file.
REPLACEMENTS = {
    # `–` U+2013 EN DASH: real UTF-8 e2 80 93 -> cp1252 "â€“" -> UTF-8 c3a2 e282ac e2809c
    b"\xc3\xa2\xe2\x82\xac\xe2\x80\x9c": "\u2013".encode("utf-8"),
    # `—` U+2014 EM DASH: real e2 80 94 -> "â€" -> c3a2 e282ac e2809d
    b"\xc3\xa2\xe2\x82\xac\xe2\x80\x9d": "\u2014".encode("utf-8"),
    # `“` U+201C: e2 80 9c -> "â€œ" -> c3a2 e282ac c593
    b"\xc3\xa2\xe2\x82\xac\xc5\x93": "\u201c".encode("utf-8"),
    # `”` U+201D: e2 80 9d -> "â€" -> c3a2 e282ac c29d
    b"\xc3\xa2\xe2\x82\xac\xc2\x9d": "\u201d".encode("utf-8"),
    # `’` U+2019: e2 80 99 -> "â€™" -> c3a2 e282ac e284a2
    b"\xc3\xa2\xe2\x82\xac\xe2\x84\xa2": "\u2019".encode("utf-8"),
}

# Stray cp1252 bytes that appear OUTSIDE a valid UTF-8 sequence.
# These are mapped char-by-char during the second pass.
STRAY_CP1252 = {0x97: "\u2014", 0x96: "\u2013", 0x92: "\u2019", 0x93: "\u201c", 0x94: "\u201d"}


def main() -> None:
    raw = PATH.read_bytes()
    out = raw
    for src, dst in REPLACEMENTS.items():
        out = out.replace(src, dst)
    # Walk bytes; keep valid UTF-8 sequences as-is, map stray cp1252 bytes.
    result = bytearray()
    i = 0
    n = len(out)
    while i < n:
        b = out[i]
        if b < 0x80:
            result.append(b)
            i += 1
            continue
        # Try to decode a UTF-8 sequence starting here.
        for size in (4, 3, 2):
            try:
                ch = out[i:i + size].decode("utf-8")
                if len(ch) == 1:
                    result.extend(out[i:i + size])
                    i += size
                    break
            except UnicodeDecodeError:
                continue
        else:
            # Not a valid UTF-8 start: treat as cp1252.
            mapped = STRAY_CP1252.get(b)
            if mapped is not None:
                result.extend(mapped.encode("utf-8"))
            # else: drop the byte silently
            i += 1
    text = bytes(result).decode("utf-8")  # verification
    PATH.write_bytes(text.encode("utf-8"))
    print("OK", len(raw), "->", PATH.stat().st_size)


if __name__ == "__main__":
    main()
