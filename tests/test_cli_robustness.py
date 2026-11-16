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


# ----------------------------------------------------- A2 / C6 local modules


@needs_cargo
def test_local_module_shadowing_a_native_stdlib_module_wins(tmp_path: Path):
    (tmp_path / "prog.py").write_text("import math\nprint(math.sqrt(16))\n", encoding="utf-8")
    first = pyrust("run", "prog.py", cwd=tmp_path)
    assert first.stdout == "4.0\n" and tier(first.stderr) == "native", first.stderr
    # A local math.py appears: Python now imports it, and so must the cached plan.
    (tmp_path / "math.py").write_text("def sqrt(x):\n    return 'local sqrt'\n", encoding="utf-8")
    expected = python("prog.py", cwd=tmp_path).stdout
    assert expected == "local sqrt\n"
    for _ in range(2):
        res = pyrust("run", "prog.py", cwd=tmp_path)
        assert res.stdout == expected, res.stderr


@needs_cargo
def test_single_entry_file_compiles_sibling_modules_and_tracks_their_edits(tmp_path: Path):
    (tmp_path / "helper.py").write_text("def cube(x):\n    return x * x * x\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("import helper\nprint(helper.cube(3))\n", encoding="utf-8")
    res = pyrust("run", "main.py", cwd=tmp_path)
    assert res.stdout == "27\n"
    assert tier(res.stderr) == "native", res.stderr
    (tmp_path / "helper.py").write_text("def cube(x):\n    return x * x * x + 1\n", encoding="utf-8")
    res = pyrust("run", "main.py", cwd=tmp_path)
    assert res.stdout == "28\n", res.stderr


# ---------------------------------------------------- A5 import-time code


def test_import_time_statements_of_helper_modules_are_not_dropped(tmp_path: Path):
    (tmp_path / "helper.py").write_text(
        'print("helper imported")\ndef f(x):\n    return x + 1\n', encoding="utf-8"
    )
    (tmp_path / "main.py").write_text("import helper\nprint(helper.f(1))\n", encoding="utf-8")
    expected = python("main.py", cwd=tmp_path).stdout
    for target in ("main.py", "."):
        res = pyrust("run", target, cwd=tmp_path)
        assert res.stdout == expected, res.stderr
        assert "runs code when imported" in res.stderr


@needs_cargo
def test_helper_module_constants_stay_native(tmp_path: Path):
    (tmp_path / "helper.py").write_text(
        '"""Helpers."""\nLIMIT = 10\n\ndef f(x):\n    return min(x, LIMIT) * 2\n'
        'if __name__ == "__main__":\n    print("not on import")\n',
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text("import helper\nprint(helper.f(30))\n", encoding="utf-8")
    res = pyrust("run", "main.py", cwd=tmp_path)
    assert res.stdout == python("main.py", cwd=tmp_path).stdout
    assert tier(res.stderr) == "native", res.stderr


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


# ------------------------------------------------------------- B1/B2 encodings


@pytest.mark.parametrize("prefix", ["", PYTHON_TIER])
def test_utf8_bom_source(tmp_path: Path, prefix: str):
    (tmp_path / "bom.py").write_bytes(b"\xef\xbb\xbf" + f"{prefix}print('hé')\n".encode("utf-8"))
    expected = python("bom.py", cwd=tmp_path)
    assert expected.returncode == 0
    res = pyrust("run", "bom.py", cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout == expected.stdout


@pytest.mark.parametrize("prefix", ["", PYTHON_TIER])
def test_pep263_latin1_source(tmp_path: Path, prefix: str):
    src = f"# -*- coding: latin-1 -*-\n{prefix}print('café')\n".encode("latin-1")
    (tmp_path / "lat.py").write_bytes(src)
    expected = python("lat.py", cwd=tmp_path)
    assert expected.stdout == "café\n"
    res = pyrust("run", "lat.py", cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout == expected.stdout


def test_undecodable_unrelated_file_in_project_is_ignored(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (proj / "junk.py").write_bytes(b"x = '\xe9\xff'\n")
    res = pyrust("run", "proj", cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout == "ok\n"


def test_undecodable_entry_reports_python_syntax_error(tmp_path: Path):
    (tmp_path / "bad.py").write_bytes(b"import collections\nx = '\xe9'\nprint(x)\n")
    expected = python("bad.py", cwd=tmp_path)
    res = pyrust("run", "bad.py", cwd=tmp_path)
    assert res.returncode == expected.returncode == 1
    assert "Traceback" not in res.stderr
    prefix = "SyntaxError: Non-UTF-8 code starting with '\\xe9'"
    assert expected.stderr.strip().splitlines()[-1].startswith(prefix)
    assert res.stderr.strip().splitlines()[-1].startswith(prefix)


# ------------------------------------------------------------- B3 cargo names


@needs_cargo
@pytest.mark.parametrize("stem", ["build", "deps"])
def test_scripts_named_like_cargo_folders_compile(tmp_path: Path, stem: str):
    (tmp_path / f"{stem}.py").write_text(NATIVE_OK, encoding="utf-8")
    res = pyrust("run", f"{stem}.py", cwd=tmp_path)
    assert res.stdout == "144\n"
    assert tier(res.stderr) == "native", res.stderr
    out = pyrust("build", f"{stem}.py", "-o", str(tmp_path / "out"), cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert f"{stem}_app" in out.stdout


# ------------------------------------------------------- B4 deleted library


@needs_cargo
def test_deleted_hybrid_library_is_rebuilt(tmp_path: Path):
    src = (
        f"{PYTHON_TIER}def work(n):\n    t = 0\n    for i in range(n):\n        t += i * i\n    return t\n"
        "print(work(1000))\n"
    )
    (tmp_path / "hy.py").write_text(src, encoding="utf-8")
    first = pyrust("run", "hy.py", cwd=tmp_path)
    assert tier(first.stderr) == "hybrid", first.stderr
    libs = [p for p in (tmp_path / ".pyrust_cache").rglob("*") if p.suffix in (".so", ".dylib", ".dll")]
    libs = [p for p in libs if p.parent.name == "release"]
    assert libs
    for lib in libs:
        lib.unlink()
    second = pyrust("run", "-q", "hy.py", cwd=tmp_path)
    assert second.stdout == first.stdout
    assert "unavailable" not in second.stderr
    assert all(lib.exists() for lib in libs)


# ------------------------------------------------- B5 / C1 build failures

FAKE_CARGO = """#!/bin/sh
if [ "$1" = "--version" ] || [ "$1" = "-V" ]; then exec "{real}" "$@"; fi
if [ -f "{flag}" ]; then
  cat "{flag}" >&2
  exit 101
fi
exec "{real}" "$@"
"""


def _fake_cargo(tmp_path: Path, message: str) -> tuple[dict[str, str], Path]:
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    flag = tmp_path / "cargo_fails"
    flag.write_text(message, encoding="utf-8")
    script = bindir / "cargo"
    script.write_text(FAKE_CARGO.format(real=find_cargo_executable(), flag=flag), encoding="utf-8")
    script.chmod(0o755)
    env = _env(PATH=os.pathsep.join([str(bindir), os.environ.get("PATH", "")]))
    return env, flag


@needs_cargo
@posix_only
def test_environment_build_failure_is_not_cached(tmp_path: Path):
    env, flag = _fake_cargo(tmp_path, "error: linker `cc` not found\n")
    (tmp_path / "p.py").write_text(NATIVE_OK, encoding="utf-8")
    res = pyrust("run", "p.py", cwd=tmp_path, env=env)
    assert res.stdout == "144\n"
    assert tier(res.stderr) == "python"
    assert "linker `cc` not found" in res.stderr
    flag.unlink()  # the machine is fixed
    res = pyrust("run", "p.py", cwd=tmp_path, env=env)
    assert res.stdout == "144\n"
    assert tier(res.stderr) == "native", res.stderr


# ------------------------------------------------------ B6 unrelated files


def test_unrelated_broken_file_does_not_break_directory_run(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "main.py").write_text(NATIVE_OK, encoding="utf-8")
    (proj / "broken.py").write_text("def oops(:\n", encoding="utf-8")
    (proj / "other.py").write_text("import collections\n", encoding="utf-8")
    res = pyrust("run", "proj", cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout == "144\n"
    if HAS_CARGO:
        assert tier(res.stderr) == "native", res.stderr


def test_fallback_runs_entry_even_if_a_reachable_module_is_broken(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "main.py").write_text("print('start')\nimport broken\n", encoding="utf-8")
    (proj / "broken.py").write_text("def oops(:\n", encoding="utf-8")
    expected = python("main.py", cwd=proj)
    res = pyrust("run", "proj", cwd=tmp_path)
    assert res.stdout == expected.stdout
    assert res.returncode == expected.returncode
    assert res.stderr.strip().splitlines()[-1] == expected.stderr.strip().splitlines()[-1]


# ------------------------------------------------------------- C4 __main__


def test_main_module_attributes_match_python(tmp_path: Path):
    probe = (
        f"{PYTHON_TIER}import sys, builtins\n"
        "print(type(__builtins__).__name__, __builtins__ is builtins)\n"
        "print(__builtins__.abs(-3))\n"
        "print(type(__loader__).__name__, __spec__, __package__, __cached__)\n"
        "print(sorted(k for k in globals() if k.startswith('__')))\n"
        "print(__file__ == sys.argv[0] or __file__.endswith('prog.py'))\n"
        "import pickle\n"
        "class P: pass\n"
        "print(type(pickle.loads(pickle.dumps(P()))).__name__)\n"
    )
    (tmp_path / "prog.py").write_text(probe, encoding="utf-8")
    expected = python("prog.py", cwd=tmp_path).stdout
    assert pyrust("run", "-q", "prog.py", cwd=tmp_path).stdout == expected


def test_uncaught_exceptions_go_through_sys_excepthook(tmp_path: Path):
    prog = (
        f"{PYTHON_TIER}import sys\n"
        "def hook(t, v, tb):\n    print('HOOK', t.__name__, v, tb.tb_frame.f_code.co_filename.endswith('e.py'))\n"
        "sys.excepthook = hook\n"
        "raise ValueError('boom')\n"
    )
    (tmp_path / "e.py").write_text(prog, encoding="utf-8")
    expected = python("e.py", cwd=tmp_path)
    res = pyrust("run", "-q", "e.py", cwd=tmp_path)
    assert res.stdout == expected.stdout == "HOOK ValueError boom True\n"
    assert res.returncode == expected.returncode == 1


def test_syntax_error_in_cpython_tier_looks_like_python(tmp_path: Path):
    (tmp_path / "s.py").write_text("x = (\n", encoding="utf-8")
    expected = python("s.py", cwd=tmp_path)
    res = pyrust("run", "-q", "s.py", cwd=tmp_path)
    assert res.returncode == expected.returncode == 1
    assert "Traceback" not in res.stderr
    assert res.stderr.strip().splitlines()[-1] == expected.stderr.strip().splitlines()[-1]


# ------------------------------------------------------------ C7 cache size


@needs_cargo
def test_superseded_builds_are_pruned(tmp_path: Path):
    script = tmp_path / "grow.py"
    for i in range(3):
        script.write_text(f"def sq(n):\n    return n * n + {i}\nprint(sq(12))\n", encoding="utf-8")
        res = pyrust("run", "grow.py", cwd=tmp_path)
        assert res.stdout == f"{144 + i}\n"
    cache = tmp_path / ".pyrust_cache"
    crates = [p for p in cache.iterdir() if p.is_dir() and p.name.startswith("grow_")]
    plans = [p for p in cache.iterdir() if p.name.startswith("plan_grow.py_")]
    assert len(crates) == 1, crates
    assert len(plans) == 1, plans


@needs_cargo
def test_pruning_spares_other_programs_with_the_same_stem(tmp_path: Path):
    (tmp_path / "twin.py").write_text(NATIVE_OK, encoding="utf-8")
    (tmp_path / "twin.pyr").write_text(NATIVE_OK.replace("12", "13"), encoding="utf-8")
    assert pyrust("run", "twin.py", cwd=tmp_path).stdout == "144\n"
    assert pyrust("run", "twin.pyr", cwd=tmp_path).stdout == "169\n"
    t0 = time.time()
    assert pyrust("run", "twin.py", cwd=tmp_path).stdout == "144\n"
    assert time.time() - t0 < 3  # still cached: not rebuilt


# ---------------------------------------------------- C10 -o safety


def test_transpile_refuses_to_overwrite_foreign_crate(tmp_path: Path):
    (tmp_path / "p.py").write_text(NATIVE_OK, encoding="utf-8")
    out = tmp_path / "mycrate"
    (out / "src").mkdir(parents=True)
    (out / "Cargo.toml").write_text("[package]\nname='mine'\n", encoding="utf-8")
    (out / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    for cmd in ("transpile", "build"):
        res = pyrust(cmd, "p.py", "-o", str(out), cwd=tmp_path)
        assert res.returncode == 1
        assert "refusing to overwrite" in res.stderr
        assert (out / "src" / "main.rs").read_text(encoding="utf-8") == "fn main() {}\n"
    # A crate PyRust generated itself may be regenerated.
    gen = tmp_path / "gen"
    assert pyrust("transpile", "p.py", "-o", str(gen), cwd=tmp_path).returncode == 0
    assert pyrust("transpile", "p.py", "-o", str(gen), cwd=tmp_path).returncode == 0


def test_output_path_that_is_a_file_gives_clean_error(tmp_path: Path):
    (tmp_path / "p.py").write_text(NATIVE_OK, encoding="utf-8")
    (tmp_path / "afile").write_text("x", encoding="utf-8")
    res = pyrust("transpile", "p.py", "-o", "afile", cwd=tmp_path)
    assert res.returncode == 1
    assert "Traceback" not in res.stderr
    assert "not a directory" in res.stderr


# ---------------------------------------------- B8 / C11 caching without cargo


def test_python_tier_plan_is_cached_without_cargo_and_dropped_when_cargo_appears(tmp_path: Path):
    (tmp_path / "p.py").write_text(NATIVE_OK, encoding="utf-8")
    env = no_cargo_env(tmp_path)
    res = pyrust("run", "p.py", cwd=tmp_path, env=env)
    assert res.stdout == "144\n" and tier(res.stderr) == "python", res.stderr
    assert "cargo" in res.stderr
    plans = list((tmp_path / ".pyrust_cache").glob("plan_p.py_*.bin"))
    assert len(plans) == 1
    # Warm: served from the cache (the transpiler is never imported).
    probe = pyrust("run", "p.py", cwd=tmp_path, env=dict(env, PYTHONVERBOSE="1"))
    assert probe.stdout == "144\n"
    assert "pyrust.codegen" not in probe.stderr
    if HAS_CARGO:
        res = pyrust("run", "p.py", cwd=tmp_path)
        assert tier(res.stderr) == "native", res.stderr


@posix_only
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores permissions")
def test_read_only_folder_uses_the_user_cache(tmp_path: Path):
    ro = tmp_path / "ro"
    ro.mkdir()
    (ro / "p.py").write_text(NATIVE_OK, encoding="utf-8")
    (ro / "q.py").write_text(f"{PYTHON_TIER}print('q')\n", encoding="utf-8")
    cache = tmp_path / "xdg"
    env = _env(XDG_CACHE_HOME=str(cache))
    ro.chmod(0o555)
    try:
        for _ in range(2):
            res = pyrust("run", "p.py", cwd=ro, env=env)
            assert res.returncode == 0 and res.stdout == "144\n", res.stderr
            assert "PermissionError" not in res.stderr
            if HAS_CARGO:
                assert tier(res.stderr) == "native", res.stderr
        res = pyrust("run", "q.py", cwd=ro, env=env)
        assert res.stdout == "q\n"
        assert list(cache.rglob("plan_p.py_*.bin")) and list(cache.rglob("plan_q.py_*.bin"))
        if HAS_CARGO:
            res = pyrust("run", "--native-only", "p.py", cwd=ro, env=env)
            assert res.returncode == 0 and res.stdout == "144\n", res.stderr
        assert not (ro / ".pyrust_cache").exists()
    finally:
        ro.chmod(0o755)
