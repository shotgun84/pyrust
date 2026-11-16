"""Unicode character-class tables taken from the running CPython.

Native code must classify characters exactly the way this interpreter does
(`str.isspace`, `str.isprintable`, `str.isdigit`, ...), so instead of
approximating with Rust's own Unicode tables we ask CPython once and emit the
answer as sorted code-point ranges. The scan is ~0.5 s, so results are cached
per Unicode version in the user's cache directory.
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata

# name -> predicate over a one-character string
PREDICATES = {
    "printable": str.isprintable,
    "space": str.isspace,
    "digit": str.isdigit,
    "decimal": str.isdecimal,
    "numeric": str.isnumeric,
    "alpha": str.isalpha,
}

_memo: dict[str, list[tuple[int, int]]] | None = None


def _cache_file() -> str:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    tag = f"{unicodedata.unidata_version}-py{sys.version_info[0]}{sys.version_info[1]}"
    return os.path.join(base, "pyrust", f"unicode-{tag}.json")


def _scan(pred) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for cp in range(0x110000):
        ok = pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def tables() -> dict[str, list[tuple[int, int]]]:
    """Ranges of code points for which each predicate is true."""
    global _memo
    if _memo is not None:
        return _memo
    path = _cache_file()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if set(data) == set(PREDICATES):
            _memo = {k: [tuple(r) for r in v] for k, v in data.items()}
            return _memo
    except (OSError, ValueError):
        pass
    _memo = {name: _scan(pred) for name, pred in PREDICATES.items()}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_memo, fh)
        os.replace(tmp, path)
    except OSError:
        pass
    return _memo


def rust_tables() -> str:
    """Rust `static` definitions for every table (ranges are inclusive)."""
    out = []
    for name, ranges in tables().items():
        body = ",".join(f"({lo},{hi})" for lo, hi in ranges)
        out.append(f"pub static PY_UCD_{name.upper()}: &[(u32, u32)] = &[{body}];")
    return "\n".join(out) + "\n"
