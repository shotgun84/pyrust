"""Robustness of the `pyrust` command: caching, project layout, encodings, launching.

Most tests drive the real command line in a subprocess (`python -m pyrust.cli`),
which is how users run it: the launcher may exec the native binary and resets
the module table before a CPython-tier program starts.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pyrust.builder import find_cargo_executable

HAS_CARGO = find_cargo_executable() is not None
needs_cargo = pytest.mark.skipif(not HAS_CARGO, reason="cargo not installed")
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX signals / permissions")

NATIVE_OK = "def sq(n):\n    return n * n\nprint(sq(12))\n"
PYTHON_TIER = "import collections\n"  # no native equivalent: forces the CPython/hybrid tier


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYRUST_QUIET", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra)
    return env


def pyrust(*args: str, cwd: Path, env: dict | None = None, stdin: str = "", timeout: int = 600):
    return subprocess.run(
        [sys.executable, "-m", "pyrust.cli", *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env or _env(),
        timeout=timeout,
    )


def python(*args: str, cwd: Path, env: dict | None = None):
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env or _env(),
        timeout=600,
    )


def tier(stderr: str) -> str:
    if "[pyrust] running on CPython + " in stderr:
        return "hybrid"
    if "[pyrust] running on CPython" in stderr:
        return "python"
    return "native"


def no_cargo_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    home = tmp_path / "home_nocargo"
    home.mkdir(exist_ok=True)
    return _env(PATH=os.pathsep.join(["/usr/bin", "/bin"]), HOME=str(home), **extra)


# --------------------------------------------------------------------- A1 migrate


def test_migrate_never_overwrites_existing_pyr(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("print('new')\n", encoding="utf-8")
    (proj / "a.pyr").write_text("print('hand edited')\n", encoding="utf-8")
    (proj / "b.py").write_text("print('b')\n", encoding="utf-8")
    res = pyrust("migrate", str(proj), cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert (proj / "a.pyr").read_text(encoding="utf-8") == "print('hand edited')\n"
    assert (proj / "b.pyr").exists()
    assert "Migrated 1 file(s)" in res.stdout
    assert "a.pyr already exists" in res.stdout

    res = pyrust("migrate", "--rename", str(proj), cwd=tmp_path)
    assert res.returncode == 0
    assert (proj / "a.pyr").read_text(encoding="utf-8") == "print('hand edited')\n"
    assert (proj / "a.py").exists()  # not renamed over the existing .pyr


def test_migrate_skips_environments_vcs_and_caches(tmp_path: Path):
    proj = tmp_path / "proj"
    for sub in (".venv/lib", "venv", "node_modules/x", ".git/hooks", "__pycache__", ".pyrust_cache", "myenv/lib"):
        (proj / sub).mkdir(parents=True)
        (proj / sub / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (proj / "myenv" / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    (proj / "lib" / "site-packages" / "dep").mkdir(parents=True)
    (proj / "lib" / "site-packages" / "dep" / "m.py").write_text("x = 1\n", encoding="utf-8")
    (proj / "src").mkdir()
    (proj / "src" / "real.py").write_text("x = 1\n", encoding="utf-8")
    res = pyrust("migrate", str(proj), cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    created = sorted(p.relative_to(proj).as_posix() for p in proj.rglob("*.pyr"))
    assert created == ["src/real.pyr"]


def test_migrate_rename_leaves_packages_importable(tmp_path: Path):
    proj = tmp_path / "proj"
    (proj / "pkg").mkdir(parents=True)
    (proj / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (proj / "pkg" / "mod.py").write_text("X = 5\n", encoding="utf-8")
    (proj / "app.py").write_text("from pkg import mod\nprint(mod.X)\n", encoding="utf-8")
    res = pyrust("migrate", "--rename", str(proj), cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert (proj / "pkg" / "mod.py").exists()
    assert (proj / "app.pyr").exists() and not (proj / "app.py").exists()
    out = pyrust("run", "-q", str(proj / "app.pyr"), cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout == "5\n"


def test_migrate_missing_directory_is_an_error(tmp_path: Path):
    res = pyrust("migrate", str(tmp_path / "nope"), cwd=tmp_path)
    assert res.returncode != 0
    assert "does not exist" in res.stderr
    assert "Traceback" not in res.stderr


# ------------------------------------------------------ D plan cache format


@pytest.mark.parametrize(
    "bad",
    [
        {"tier": "native", "binary": None},
        {"tier": "native"},
        {"tier": "hybrid", "script": 5},
        {"tier": "python", "script": None},
        {"tier": "python", "script": "x", "names": 3},
        ["not", "a", "dict"],
    ],
)
def test_malformed_plan_is_discarded(tmp_path: Path, bad):
    import marshal

    (tmp_path / "p.py").write_text(f"{PYTHON_TIER}print('fine')\n", encoding="utf-8")
    env = _env()
    first = pyrust("run", "p.py", cwd=tmp_path, env=env)
    assert first.stdout == "fine\n"
    plans = list((tmp_path / ".pyrust_cache").glob("plan_*.bin"))
    assert len(plans) == 1
    plans[0].write_bytes(marshal.dumps(bad))
    res = pyrust("run", "p.py", cwd=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    assert res.stdout == "fine\n"


def test_toolchain_identity_is_part_of_the_cache_key(tmp_path: Path):
    code = "from pyrust.plancache import toolchain_salt; print(toolchain_salt())"
    with_cargo = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=_env())
    without = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=no_cargo_env(tmp_path)
    )
    assert with_cargo.returncode == without.returncode == 0
    if HAS_CARGO:
        assert with_cargo.stdout != without.stdout


# ------------------------------------------------------------ A6 entry point


def test_directory_entry_matches_python(tmp_path: Path):
    proj = tmp_path / "d1"
    proj.mkdir()
    (proj / "__main__.py").write_text(f"{PYTHON_TIER}print('dunder main')\n", encoding="utf-8")
    (proj / "app.py").write_text("print('app')\n", encoding="utf-8")
    expected = python("d1", cwd=tmp_path)
    assert expected.stdout == "dunder main\n"
    res = pyrust("run", "d1", cwd=tmp_path)
    assert res.stdout == expected.stdout, res.stderr


def test_directory_entry_fallback_is_alphabetical_across_extensions(tmp_path: Path):
    from pyrust.parser import find_entry

    (tmp_path / "zeta.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "beta.pyr").write_text("print(2)\n", encoding="utf-8")
    assert find_entry(tmp_path).name == "beta.pyr"
    (tmp_path / "beta.py").write_text("print(3)\n", encoding="utf-8")
    assert find_entry(tmp_path).name == "beta.pyr"  # .pyr wins on a stem tie, like the import hook
