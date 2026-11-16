"""Differential tests: every program must behave identically under `python` and `pyrust run`.

Each case is a `.py` file under tests/differential/<group>/, or a directory there
whose entry point is `main.py`. The program runs once with the current Python
interpreter and once through PyRust (cold build, then warm from the cache), and
the test compares:

* stdout, byte for byte;
* the exit status;
* when Python exits with an error, the last line of stderr (the exception line).

Optional header comments in the entry file tune a case:

    # pyrust: tier=native       require the tier PyRust actually used (native|hybrid|python)
    # pyrust: tier!=native      forbid a tier (e.g. a construct native code cannot do correctly)
    # pyrust: stdin=<text>      feed this text (\\n escapes allowed) on stdin
    # pyrust: args=a b c        command-line arguments
    # pyrust: stderr=any        do not compare the exception line

Native-tier cases need Rust; without cargo they are skipped.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
CASES_DIR = ROOT / "differential"
HAS_CARGO = shutil.which("cargo") is not None


def _collect() -> list[Path]:
    cases: list[Path] = []
    for group in sorted(p for p in CASES_DIR.iterdir() if p.is_dir()):
        for entry in sorted(group.iterdir()):
            if entry.is_dir() and (entry / "main.py").exists():
                cases.append(entry)
            elif entry.suffix == ".py":
                cases.append(entry)
    return cases


def _directives(source: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"^#\s*pyrust:\s*(\S+?)\s*(!=|=)\s*(.*)$", source, re.M):
        key, op, value = m.group(1), m.group(2), m.group(3).strip()
        out[key + ("!" if op == "!=" else "")] = value
    return out


def _actual_tier(stderr: str) -> str:
    if "[pyrust] running on CPython + " in stderr:
        return "hybrid"
    if "[pyrust] running on CPython" in stderr:
        return "python"
    return "native"


def _strip_notes(stderr: str) -> str:
    return "\n".join(line for line in stderr.splitlines() if not line.startswith("[pyrust]"))


def _last_line(stderr: str) -> str:
    lines = [line for line in _strip_notes(stderr).splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""


def _run(cmd: list[str], cwd: Path, stdin: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=stdin.encode("utf-8"),
        capture_output=True,
        timeout=600,
        env=env,
    )


CASES = _collect()


@pytest.mark.parametrize("case", CASES, ids=[f"{c.parent.name}/{c.name}" for c in CASES])
def test_matches_cpython(case: Path, tmp_path: Path) -> None:
    work = tmp_path / "prog"
    if case.is_dir():
        shutil.copytree(case, work)
        entry = work / "main.py"
    else:
        work.mkdir()
        entry = work / case.name
        shutil.copy2(case, entry)
    source = entry.read_text(encoding="utf-8", errors="replace")
    opts = _directives(source)
    if not HAS_CARGO and (opts.get("tier") in ("native", "hybrid")):
        pytest.skip("cargo not installed")
    stdin = opts.get("stdin", "").encode().decode("unicode_escape")
    args = opts.get("args", "").split()

    env = dict(os.environ)
    env.pop("PYRUST_QUIET", None)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"

    expected = _run([sys.executable, entry.name, *args], work, stdin, env)
    pyrust = [sys.executable, "-m", "pyrust.cli", "run", entry.name, *args]
    cold = _run(pyrust, work, stdin, env)
    warm = _run(pyrust, work, stdin, env)

    for label, got in (("cold", cold), ("warm", warm)):
        tier = _actual_tier(got.stderr.decode("utf-8", "replace"))
        # Generated Rust that fails to compile is a PyRust bug even though the
        # program still runs (on CPython): it must never pass silently.
        assert "Cargo build failed" not in got.stderr.decode("utf-8", "replace"), (
            f"native build failed\n{got.stderr.decode('utf-8', 'replace')[-3000:]}"
        )
        context = (
            f"[{label} run, tier={tier}]\n--- pyrust stderr ---\n"
            f"{got.stderr.decode('utf-8', 'replace')[-3000:]}\n"
            f"--- python stderr ---\n{expected.stderr.decode('utf-8', 'replace')[-1500:]}"
        )
        assert got.stdout.decode("utf-8", "replace") == expected.stdout.decode("utf-8", "replace"), context
        assert got.returncode == expected.returncode, context
        if expected.returncode != 0 and opts.get("stderr") != "any":
            assert _last_line(got.stderr.decode("utf-8", "replace")) == _last_line(
                expected.stderr.decode("utf-8", "replace")
            ), context
        if "tier" in opts:
            assert tier == opts["tier"], f"expected tier {opts['tier']}, got {tier}\n{context}"
        if "tier!" in opts:
            assert tier != opts["tier!"], f"tier {tier} is forbidden here\n{context}"
