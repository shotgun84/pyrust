"""Tiered execution: native -> hybrid (CPython + native functions) -> CPython."""

import subprocess
import sys
from pathlib import Path

import pytest

from pyrust.builder import find_cargo_executable, plan_execution, run_project
from pyrust.cli import main
from pyrust.errors import PyRustError
from pyrust.hybrid import plan_hybrid
from pyrust.hybrid_runner import instrument

needs_cargo = pytest.mark.skipif(find_cargo_executable() is None, reason="cargo not installed")

MIXED = '''
import re
import math

MOD = 1000000007
calls = []


def fact(n):
    return 1 if n <= 1 else n * fact(n - 1)


def powmod(b, e):
    r = 1
    for _ in range(e):
        r = r * b % MOD
    return r


def ratio_sum(a, n):
    t = 0.0
    for i in range(n):
        t += a / (i - 3)
    return t


def root_sum(x: float, n: int):
    t = 0.0
    for i in range(n):
        t += math.sqrt(x - i)
    return t


def half(a):
    return a / 2


def noisy(n):
    print("noisy", n)
    return n


def tracked(n):
    calls.append(n)
    return n


def guarded(n):
    try:
        return 10 // n
    except ZeroDivisionError:
        return -1


def show(fn, *args, **kw):
    try:
        r = fn(*args, **kw)
        print(type(r).__name__, repr(r))
    except Exception as exc:
        print("EXC", type(exc).__name__, exc)


show(fact, 20)
show(fact, 30)          # overflows i64: must fall back to the exact bigint
show(fact, 5.0)         # wrong runtime type: must fall back
show(powmod, 3, 5000)
show(powmod, b=3, e=10)
show(ratio_sum, 1, 3)
show(ratio_sum, 1, 9)   # hits a zero divisor: must raise ZeroDivisionError, not return inf
show(root_sum, 9.0, 4)
show(root_sum, 2.0, 9)  # sqrt of a negative: must raise ValueError, not return nan
show(root_sum, x=9.0, n=2)
show(half, 7)
show(noisy, 3)
show(tracked, 4)
show(guarded, 0)
print(re.sub("a", "b", "banana"), calls, fact.__name__)
for i in range(3):
    show(powmod, i, 10)
'''


def test_plan_selects_only_pure_scalar_functions():
    plan = plan_hybrid(MIXED, "mixed.py")
    assert set(plan.names) == {"fact", "powmod", "ratio_sum", "root_sum"}
    sigs = {f.name: (f.params, f.ret) for f in plan.functions}
    assert sigs["fact"] == (["int"], "int")
    assert sigs["ratio_sum"] == (["int", "int"], "float")
    assert sigs["root_sum"] == (["float", "int"], "float")
    assert "too small" in plan.skipped["half"]  # no loop: the bridge would cost more than it saves
    assert "print" in plan.skipped["noisy"]
    assert "calls.append" in plan.skipped["tracked"]
    assert "Try" in plan.skipped["guarded"]
    assert 'extern "C"' not in plan.rust_source  # wrappers live in lib.rs, not the module
    assert "pub fn fact(n: i64) -> i64" in plan.rust_source


def test_plan_rejects_rebound_and_decorated_functions():
    src = (
        "import functools\n"
        "def a(n):\n    return n + 1\n"
        "def a(n):\n    return n + 2\n"
        "@functools.cache\ndef b(n):\n    return n * 2\n"
        "def c(n):\n    global K\n    K = n\n    return n\n"
        "def d(n):\n    t = 0\n    for i in range(n):\n        t += i << 70\n    return t\n"
        "print(a(1), b(2), c(3), d(1))\n"
    )
    plan = plan_hybrid(src, "x.py")
    assert plan.names == []
    assert set(plan.skipped) == {"a", "b", "c", "d"}


def test_instrument_rebinds_only_named_functions():
    src = "def f(n):\n    return n\n\ndef g(n):\n    return n\n\nout = (f(1), g(1))\n"
    ns = {"__pyrust_accel__": {"f": lambda orig: (lambda n: orig(n) + 100)}}
    exec(instrument(src, "x.py", {"f"}), ns)
    assert ns["out"] == (101, 1)


def test_unsupported_program_falls_back_to_cpython(tmp_path: Path, capfd):
    script = tmp_path / "plain.pyr"
    script.write_text(
        "import collections, sys\n"
        "c = collections.Counter('hello')\n"
        "try:\n    print(c.most_common(1), sys.argv[1:])\nexcept Exception:\n    pass\n",
        encoding="utf-8",
    )
    assert plan_execution(script, build=False).tier == "python"
    rc = run_project(script, args=["a", "b"])
    out, err = capfd.readouterr()
    assert rc == 0
    assert out.strip() == "[('l', 2)] ['a', 'b']"
    assert "running on CPython" in err

    with pytest.raises(PyRustError):
        run_project(script, native_only=True)


def test_cpython_fallback_propagates_exit_code_and_imports_pyr_modules(tmp_path: Path, capfd):
    (tmp_path / "helper.pyr").write_text("def twice(x):\n    return [x] * 2\n", encoding="utf-8")
    script = tmp_path / "main.pyr"
    script.write_text(
        "import sys, helper\nprint(helper.twice(lambda: 1)[0]())\nsys.exit(7)\n", encoding="utf-8"
    )
    rc = run_project(script, quiet=True)
    out, err = capfd.readouterr()
    assert rc == 7
    assert out.strip() == "1"
    assert err == ""


def test_check_command_reports_tiers(tmp_path: Path, capsys):
    script = tmp_path / "mixed.pyr"
    script.write_text(MIXED, encoding="utf-8")
    assert main(["check", str(script)]) == 0
    out = capsys.readouterr().out
    assert "HYBRID" in out and "native  fact(int) -> int" in out and "python  noisy" in out

    native = tmp_path / "pure.pyr"
    native.write_text("def sq(n):\n    return n * n\nprint(sq(12))\n", encoding="utf-8")
    assert main(["check", str(native)]) == 0
    assert "NATIVE" in capsys.readouterr().out


@needs_cargo
def test_hybrid_output_is_identical_to_cpython(tmp_path: Path, capfd):
    script = tmp_path / "mixed.pyr"
    script.write_text(MIXED, encoding="utf-8")
    expected = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    ).stdout

    plan = plan_execution(script)
    assert plan.tier == "hybrid"
    assert set(plan.native_functions) == {"fact", "powmod", "ratio_sum", "root_sum"}

    rc = run_project(script)
    out, err = capfd.readouterr()
    assert rc == 0
    assert out.replace("\r\n", "\n") == expected.replace("\r\n", "\n")
    assert "4 native function(s)" in err

    # Second run reuses the cached library (no re-planning).
    assert plan_execution(script).manifest_path == plan.manifest_path


@needs_cargo
def test_native_tier_aborts_on_int_overflow_instead_of_wrapping(tmp_path: Path, capfd):
    script = tmp_path / "ovf.pyr"
    script.write_text(
        "def grow(n):\n    x = 1\n    for i in range(n):\n        x = x * 3\n    return x\n"
        "print(grow(10))\nprint(grow(80))\n",
        encoding="utf-8",
    )
    rc = run_project(script, native_only=True)
    out, _ = capfd.readouterr()
    assert out.split() == ["59049"]  # correct first result, then a loud abort
    assert rc != 0


def test_constant_bigint_is_refused_natively():
    from pyrust import transpile_source

    with pytest.raises(PyRustError):
        transpile_source("x = 2 ** 100\nprint(x)\n")
