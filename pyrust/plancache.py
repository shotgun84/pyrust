"""Remembers how a program runs so repeat launches skip all analysis.

Imports only cheap stdlib modules on purpose: `pyrust run` on an unchanged
program should cost little more than starting Python itself.
"""

from __future__ import annotations

import hashlib
import marshal  # built in: avoids the ~15 ms `json` import on every launch
import os
import sys

_EXTS = (".py", ".pyr")
_salt: str | None = None


def toolchain_salt() -> str:
    """Changes whenever PyRust itself (or the host Python) changes."""
    global _salt
    if _salt is None:
        digest = hashlib.sha256()
        here = os.path.dirname(os.path.abspath(__file__))
        for name in sorted(os.listdir(here)):
            if name.endswith(".py"):
                with open(os.path.join(here, name), "rb") as fh:
                    digest.update(fh.read())
        digest.update(sys.version.encode())
        _salt = digest.hexdigest()[:12]
    return _salt


def _source_files(target: str) -> tuple[str, list[str]]:
    """(cache base dir, every source file that can influence the plan)."""
    if os.path.isdir(target):
        base = target
        files = [os.path.join(base, n) for n in os.listdir(base) if n.endswith(_EXTS)]
    else:
        base = os.path.dirname(target)
        files = [target]
    vendor = os.path.join(base, "pyrust_vendor")
    if os.path.isdir(vendor):
        files += [os.path.join(vendor, n) for n in os.listdir(vendor) if n.endswith(_EXTS)]
    return base, sorted(files)


def _plan_path(target: str, release: bool) -> str:
    target = os.path.abspath(target)
    base, files = _source_files(target)
    digest = hashlib.sha256()
    digest.update(f"{toolchain_salt()}\0{int(release)}\0{target}\0".encode())
    for path in files:
        digest.update(os.path.basename(path).encode() + b"\0")
        with open(path, "rb") as fh:
            digest.update(fh.read())
        digest.update(b"\0")
    stem = os.path.splitext(os.path.basename(target))[0] or "app"
    return os.path.join(base, ".pyrust_cache", f"plan_{stem}_{digest.hexdigest()[:16]}.bin")


def load_plan(target: str, release: bool = True) -> dict | None:
    """The cached plan for `target`, or None if there is none / it went stale."""
    try:
        with open(_plan_path(target, release), "rb") as fh:
            plan = marshal.load(fh)
        if not isinstance(plan, dict):
            return None
        for key in ("binary", "script", "manifest_path", "dll"):
            if plan.get(key) and not os.path.exists(plan[key]):
                return None
        if plan.get("tier") not in ("native", "hybrid", "python"):
            return None
        return plan
    except (OSError, ValueError, EOFError, TypeError):
        return None


def save_plan(target: str, plan: dict, release: bool = True) -> None:
    try:
        path = _plan_path(target, release)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            marshal.dump(plan, fh)
    except (OSError, ValueError):
        pass  # a read-only folder just means no caching
