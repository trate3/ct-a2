"""Remove account names, host names and file-system paths from shipped data files.

Generic patterns only: absolute home, scratch and user directories become `<HOME>`,
`<SCRATCH>`; cluster node names become `<NODE>`; user, group and account names that
earlier passes had replaced by placeholders become neutral words. Files are edited in
place, including the contents of `.gz` files. Hashes recorded inside the files were
computed on the original bytes and no longer match a recomputation.

Usage: python tools/redact.py DIR [DIR ...]
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

RULES: list[tuple[re.Pattern, bytes]] = [
    (re.compile(rb"/home/[A-Za-z0-9_.\-]+"), b"<HOME>"),
    (re.compile(rb"/Users/[A-Za-z0-9_.\-]+"), b"<HOME>"),
    (re.compile(rb"/scratch/[A-Za-z0-9_.\-]+"), b"<SCRATCH>"),
    (re.compile(rb"\bgrp-user0\d\b"), b"group"),
    (re.compile(rb"\buser0\d\b"), b"user"),
    (re.compile(rb"\bacct0\b"), b"<ACCOUNT>"),
    (re.compile(rb"\bcluster-nd-\d+\b"), b"<NODE>"),
]


def redact(data: bytes) -> bytes:
    for rx, new in RULES:
        data = rx.sub(new, data)
    return data


def main(dirs: list[str]) -> int:
    changed = 0
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if not p.is_file():
                continue
            raw = p.read_bytes()
            if p.suffix == ".gz":
                inner = gzip.decompress(raw)
                new = redact(inner)
                if new != inner:
                    p.write_bytes(gzip.compress(new, mtime=0))
                    changed += 1
            else:
                new = redact(raw)
                if new != raw:
                    p.write_bytes(new)
                    changed += 1
    print(f"redacted {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
