"""Remembers how a program runs so repeat launches skip all analysis.

Imports only cheap stdlib modules on purpose: `pyrust run` on an unchanged
program should cost little more than starting Python itself. (Whatever this
module imports is dropped from `sys.modules` again before the user's program
runs, see `pyrust.restore_modules`.)
"""

from __future__ import annotations

import hashlib
import marshal  # built in: avoids the ~15 ms `json` import on every launch
import os
import sys

_EXTS = (".py", ".pyr")
_TIERS = ("native", "hybrid", "python")
_salt: str | None = None
_toolchain: str | None = None


def find_tool(name: str) -> str | None:
    """Locate a Rust tool (`cargo`, `rustc`) on PATH or in ~/.cargo/bin, using only os calls."""
    exts = [""]
    if os.name == "nt":
        exts = [e.lower() for e in os.environ.get("PATHEXT", ".EXE;.CMD;.BAT").split(os.pathsep) if e] or [".exe"]
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        if not folder:
            continue
        for ext in exts:
            cand = os.path.join(folder, name + ext)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    home_bin = os.path.join(os.path.expanduser("~"), ".cargo", "bin", name + (".exe" if os.name == "nt" else ""))
    if os.path.isfile(home_bin):
        return home_bin
    return None


def _stat_tag(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "-"


def toolchain_fingerprint() -> str:
    """A cheap identity of the Rust toolchain (stat calls only, no subprocess).

    Changes when cargo/rustc appear, disappear, move or are updated, including
    toolchain updates behind rustup's proxies.
    """
    global _toolchain
    if _toolchain is not None:
        return _toolchain
    parts: list[str] = []
    for tool in ("cargo", "rustc"):
        path = find_tool(tool)
        if path is None:
            parts.append(f"{tool}=none")
            continue
        real = os.path.realpath(path)
        parts.append(f"{tool}={path}>{real}@{_stat_tag(real)}")
    if parts[0] != "cargo=none":
        rustup_home = os.environ.get("RUSTUP_HOME") or os.path.join(os.path.expanduser("~"), ".rustup")
        if os.path.isdir(rustup_home):
            parts.append("rustup=" + os.environ.get("RUSTUP_TOOLCHAIN", ""))
            settings = os.path.join(rustup_home, "settings.toml")
            try:
                with open(settings, "rb") as fh:
                    parts.append(hashlib.sha256(fh.read()).hexdigest()[:12])
            except OSError:
                pass
            chains = os.path.join(rustup_home, "toolchains")
            try:
                names = sorted(os.listdir(chains))
            except OSError:
                names = []
            for name in names:
                manifest = os.path.join(chains, name, "lib", "rustlib", "multirust-channel-manifest.toml")
                parts.append(f"{name}@{_stat_tag(manifest)}")
    _toolchain = "|".join(parts)
    return _toolchain


def has_cargo() -> bool:
    return not toolchain_fingerprint().startswith("cargo=none")


def toolchain_salt() -> str:
    """Changes whenever PyRust itself, the host Python or the Rust toolchain changes."""
    global _salt
    if _salt is None:
        digest = hashlib.sha256()
        here = os.path.dirname(os.path.abspath(__file__))
        for name in sorted(os.listdir(here)):
            if name.endswith(".py"):
                with open(os.path.join(here, name), "rb") as fh:
                    digest.update(fh.read())
        digest.update(sys.version.encode())
        digest.update(toolchain_fingerprint().encode("utf-8", "surrogateescape"))
        _salt = digest.hexdigest()[:12]
    return _salt


def user_cache_root() -> str:
    """Per-user cache folder used when a program's own folder is not writable."""
    home = os.path.expanduser("~")
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, "pyrust", "Cache")
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and os.path.isabs(xdg):
        return os.path.join(xdg, "pyrust")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Caches", "pyrust")
    return os.path.join(home, ".cache", "pyrust")


def cache_dir(base: str) -> str:
    """Where build artifacts and plans for programs in folder `base` live.

    `<base>/.pyrust_cache` when that is (or can be made) writable, otherwise a
    folder under the per-user cache named after `base`.
    """
    base = os.path.abspath(base)
    local = os.path.join(base, ".pyrust_cache")
    writable = os.W_OK | os.X_OK
    if os.path.isdir(local):
        if os.access(local, writable):
            return local
    elif os.access(base, writable):
        return local
    tag = hashlib.sha256(base.encode("utf-8", "surrogateescape")).hexdigest()[:16]
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in os.path.basename(base)) or "root"
    return os.path.join(user_cache_root(), f"{name[:40]}_{tag}")


def _source_files(target: str) -> tuple[str, list[str]]:
    """(cache base dir, source files that always influence the plan)."""
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


def file_digest(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:24]
    except OSError:
        return "-"


def plan_stem(target: str, release: bool = True) -> str:
    """`plan_<file name>_<r|d>`: one family of plan files per program and build profile."""
    name = os.path.basename(os.path.abspath(target)) or "app"
    name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return f"plan_{name}_{'r' if release else 'd'}"


def _plan_path(target: str, release: bool) -> str:
    target = os.path.abspath(target)
    base, files = _source_files(target)
    digest = hashlib.sha256()
    digest.update(f"{toolchain_salt()}\0{int(release)}\0{target}\0".encode("utf-8", "surrogateescape"))
    for path in files:
        digest.update(os.path.basename(path).encode("utf-8", "surrogateescape") + b"\0")
        with open(path, "rb") as fh:
            digest.update(fh.read())
        digest.update(b"\0")
    return os.path.join(cache_dir(base), f"{plan_stem(target, release)}_{digest.hexdigest()[:16]}.bin")


def _valid(plan: object) -> bool:
    """A plan read from disk must have exactly the shape `pyrust run` relies on."""
    if not isinstance(plan, dict) or plan.get("tier") not in _TIERS:
        return False
    for key in ("binary", "script", "manifest_path", "dll", "blocker", "crate_dir", "entry"):
        if plan.get(key) is not None and not isinstance(plan[key], str):
            return False
    if plan["tier"] == "native" and not plan.get("binary"):
        return False
    if plan["tier"] != "native" and not plan.get("script"):
        return False
    if plan["tier"] == "hybrid" and not plan.get("manifest_path"):
        return False
    names = plan.get("names", [])
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return False
    deps = plan.get("deps", [])
    if not isinstance(deps, list) or not all(
        isinstance(d, (list, tuple)) and len(d) == 2 and all(isinstance(x, str) for x in d) for d in deps
    ):
        return False
    absent = plan.get("absent", [])
    if not isinstance(absent, list) or not all(isinstance(p, str) for p in absent):
        return False
    return True


def load_plan(target: str, release: bool = True) -> dict | None:
    """The cached plan for `target`, or None if there is none / it went stale."""
    try:
        with open(_plan_path(target, release), "rb") as fh:
            plan = marshal.load(fh)
        if not _valid(plan):
            return None
        for key in ("binary", "script", "manifest_path", "dll"):
            if plan.get(key) and not os.path.exists(plan[key]):
                return None
        # Local modules the program imports: any edit invalidates the plan...
        for path, digest in plan.get("deps", []):
            if file_digest(path) != digest:
                return None
        # ...and so does a new local file that would shadow an imported module.
        for path in plan.get("absent", []):
            if os.path.exists(path):
                return None
        return plan
    except Exception:  # a damaged cache must never stop a program from running
        return None


def save_plan(target: str, plan: dict, release: bool = True) -> str | None:
    """Store `plan`; returns the plan file path (None when the cache is not writable)."""
    try:
        path = _plan_path(target, release)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            marshal.dump(plan, fh)
        os.replace(tmp, path)
        return path
    except (OSError, ValueError):
        return None  # an unwritable cache just means no caching
