"""Hybrid tier safety: a native fast path must never change what a program does.

The runtime tests drive the real guarded-wrapper machinery (`instrument`,
`_wrap`) with fake "native" functions written in Python, so they run without
Rust; the build tests at the end need cargo.
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

from pyrust import hybrid
from pyrust.builder import find_cargo_executable, plan_execution
from pyrust.hybrid import plan_hybrid, prepare_hybrid
from pyrust.hybrid_runner import _Context, _wrap, instrument

needs_cargo = pytest.mark.skipif(find_cargo_executable() is None, reason="cargo not installed")

I64_MIN = -(2**63)


# --------------------------------------------------------------------------
# Static analysis
# --------------------------------------------------------------------------


def test_star_import_of_local_module_counts_the_names_it_binds(tmp_path: Path):
    (tmp_path / "local_settings.py").write_text("SCALE = 10\ndef helper(x):\n    return x * 1000\n_hidden = 1\n")
    src = (
        "import collections\nSCALE = 3\n_hidden = 2\n"
        "def helper(x):\n    return x + 1\n"
        "def compute(n):\n    t = 0\n    for i in range(n):\n        t += i * SCALE + helper(i)\n    return t\n"
        "def private(n):\n    t = 0\n    for i in range(n):\n        t += _hidden\n    return t\n"
        "from local_settings import *\n"
        "print(compute(4), private(2))\n"
    )
    plan = plan_hybrid(src, str(tmp_path / "main.py"))
    assert "compute" not in plan.names
    assert plan.skipped["helper"] == "is rebound elsewhere in the module"
    assert "private" in plan.names  # `_hidden` is not exported by `import *`


def test_star_import_with_unknown_names_blocks_everything(tmp_path: Path):
    src = (
        "from not_a_real_module_xyz import *\n"
        "def f(n):\n    t = 0\n    for i in range(n):\n        t += abs(i)\n    return t\n"
        "print(f(3))\n"
    )
    plan = plan_hybrid(src, str(tmp_path / "main.py"))
    assert plan.names == []
    assert "not_a_real_module_xyz import *" in plan.skipped["f"]


def test_star_import_respects_literal_all(tmp_path: Path):
    (tmp_path / "mod.py").write_text("__all__ = ['OTHER']\nK = 5\nOTHER = 1\n")
    src = (
        "K = 2\n"
        "def f(n):\n    t = 0\n    for i in range(n):\n        t += K\n    return t\n"
        "from mod import *\nprint(f(3))\n"
    )
    assert plan_hybrid(src, str(tmp_path / "main.py")).names == ["f"]


def test_except_as_and_match_captures_are_bindings():
    src = (
        "RATE = 2\nBASE = 10\n"
        "def helper(x):\n    return x + 1\n"
        "def scaled(n):\n    t = 0\n    for i in range(n):\n        t += RATE\n    return t\n"
        "def based(n):\n    t = 0\n    for i in range(n):\n        t += BASE\n    return t\n"
        "def helped(n):\n    t = 0\n    for i in range(n):\n        t += helper(i)\n    return t\n"
        "try:\n    int('x')\nexcept ValueError as RATE:\n    pass\n"
        "match 99:\n    case BASE:\n        pass\n"
        "match [1]:\n    case [*helper]:\n        pass\n"
    )
    plan = plan_hybrid(src, "x.py")
    assert plan.names == []
    assert plan.skipped["scaled"] == "reads global 'RATE'"
    assert plan.skipped["based"] == "reads global 'BASE'"
    assert plan.skipped["helper"] == "is rebound elsewhere in the module"


def test_dependencies_are_transitive():
    src = (
        "import math\nLIMIT = 5\nHALF = 0.5\n"
        "def step(x):\n    return x + LIMIT\n"
        "def run(n):\n    t = 0\n    for i in range(n):\n        t = step(t)\n    return t\n"
        "def outer(n):\n    t = 0.0\n    for i in range(n):\n        t += run(2) * HALF + math.sqrt(abs(i))\n    return t\n"
        "def rec(n):\n    return 0 if n <= 0 else 1 + rec(n - 1)\n"
        "print(outer(3), rec(3))\n"
    )
    plan = plan_hybrid(src, "x.py")
    deps = {f.name: f.deps for f in plan.functions}
    assert deps["run"] == {"funcs": ["step"], "consts": {"LIMIT": ["int", 5]}, "math": None, "builtins": ["range"]}
    assert deps["outer"] == {
        "funcs": ["run", "step"],
        "consts": {"HALF": ["float", (0.5).hex()], "LIMIT": ["int", 5]},
        "math": ["sqrt"],
        "builtins": ["abs", "range"],
    }
    assert deps["rec"]["funcs"] == ["rec"]  # recursion: its own global name is a dependency


def test_disagreeing_call_sites_get_one_signature_each():
    src = (
        "import collections\n"
        "def fact(n):\n    r = 1\n    i = 2\n    while i <= n:\n        r = r * i\n        i += 1\n    return r\n"
        "print(fact(20), fact(20), fact(5.0))\n"
    )
    plan = plan_hybrid(src, "x.py")
    (fact,) = plan.functions
    assert [(v.params, v.ret) for v in fact.variants] == [(["int"], "int"), (["float"], "int")]
    assert (fact.params, fact.ret) == (["int"], "int")  # the most common call site first
    assert plan.variant_sources[1] and "pub fn fact(n: f64) -> i64" in plan.variant_sources[1]


def test_variant_count_is_capped():
    calls = "f(1, 1)\nf(1.0, 1)\nf(1, 1.0)\nf(1.0, 1.0)\nf(True, 1)\nf(1, True)\n"
    src = "def f(a, b):\n    t = 0\n    for i in range(3):\n        t = t + a * b\n    return t\n" + calls
    (f,) = plan_hybrid(src, "x.py").functions
    assert len(f.variants) <= hybrid.MAX_SIGNATURES


def test_literal_iteration_gives_type_hints():
    src = (
        "def fdiv(a, b):\n    r = 0.0\n    for _ in range(1):\n        r = a // b\n    return r\n"
        "nan = float('nan')\n"
        "for a, b in [(1.0, 0.1), (-7.5, 2.0), (nan, 1.0)]:\n    print(fdiv(a, b))\n"
    )
    (f,) = plan_hybrid(src, "x.py").functions
    assert f.params == ["float", "float"]


def test_unhinted_signature_is_a_reported_guess():
    src = (
        "def total(n):\n    t = 0\n    for i in range(3):\n        t = t + n\n    return t\n"
        "for f in [total]:\n    print(f(2), f(2.5))\n"
    )
    plan = plan_hybrid(src, "x.py")
    (total,) = plan.functions
    assert [v.params for v in total.variants] == [["int"], ["float"]]
    assert "guessed" in plan.notes["total"]


# --------------------------------------------------------------------------
# Runtime guard machinery (fake native functions, no Rust needed)
# --------------------------------------------------------------------------


class FakeLib:
    """Stands in for the native library: records calls, can be told to bail out."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.failed = False
        self.bail = False

    def take_error(self) -> int:
        f, self.failed = self.failed, False
        return int(f)

    def export(self, name: str, impl, ret: str):
        fail = {"int": I64_MIN, "float": float("nan"), "bool": 2}[ret]

        def native(*args):
            self.calls.append((name, *args))
            if self.bail:
                self.failed = True
                return fail
            out = impl(*args)
            return int(out) if ret == "bool" else out

        return native


def accelerate(src: str, plan_src_specs: dict, lib: FakeLib, filename: str = "/virtual/prog.py") -> dict:
    """Run `src` with every function in `plan_src_specs` swapped for a fake native one.

    plan_src_specs: name -> (deps, [(params, ret, impl), ...]); deps funcs that are
    not accelerated themselves get the identity-recording decorator.
    """
    ctx = _Context(lib.take_error)
    accel = {}
    for name, (deps, variants) in plan_src_specs.items():
        calls = [({"params": p, "ret": r}, lib.export(name, impl, r)) for p, r, impl in variants]
        spec = {"name": name, "deps": deps}
        accel[name] = lambda orig, _s=spec, _c=calls: _wrap(orig, _s, _c, ctx)
    for _name, (deps, _v) in plan_src_specs.items():
        for dep in deps.get("funcs", ()):
            accel.setdefault(dep, lambda orig, _n=dep: ctx.register(_n, orig))
    ns = {"__name__": "__main__", "__pyrust_accel__": accel}
    exec(instrument(src, filename, set(accel)), ns)
    return ns


def no_deps(**kw):
    return {"funcs": [], "consts": {}, "math": None, "builtins": [], **kw}


LOOP = "def tri(n, k=2):\n    t = 0\n    for i in range(n):\n        t += i * k\n    return t\n"


def tri_native(n, k):
    return sum(i * k for i in range(n))


def test_fast_path_only_for_exact_types():
    lib = FakeLib()
    ns = accelerate(LOOP, {"tri": (no_deps(builtins=["range"]), [(["int", "int"], "int", tri_native)])}, lib)
    tri = ns["tri"]

    class MyInt(int):
        pass

    assert tri(5) == 20 and tri(n=5, k=3) == 30 and tri(k=3, n=5) == 30
    assert len(lib.calls) == 3
    assert tri(True) == 0 and tri(5, MyInt(2)) == 20 and tri(5, 2.5) == 25.0 and tri(3, 2**70) == 3 * 2**70
    assert len(lib.calls) == 3  # bool, int subclass, float, beyond i64: all ran the Python body
    tri.__defaults__ = (10,)
    assert tri(5) == 100  # mutated defaults are seen by the fast path too
    assert lib.calls[-1] == ("tri", 5, 10)


def test_bail_out_runs_the_python_body():
    lib = FakeLib()
    ns = accelerate(LOOP, {"tri": (no_deps(builtins=["range"]), [(["int", "int"], "int", tri_native)])}, lib)
    lib.bail = True
    assert ns["tri"](5) == 20
    assert lib.calls == [("tri", 5, 2)]


def test_rebinding_any_dependency_falls_back():
    src = (
        "import math\nLIMIT = 5\n"
        "def step(x):\n    return x + 1\n"
        "def count(n):\n    c = 0\n    for i in range(n):\n        if i < LIMIT:\n            c = step(c)\n"
        "    return c + math.floor(abs(-0.5))\n"
    )
    deps = {"funcs": ["step"], "consts": {"LIMIT": ["int", 5]}, "math": ["floor"], "builtins": ["abs", "range"]}

    def native(n):
        return min(n, 5)

    lib = FakeLib()
    ns = accelerate(src, {"count": (deps, [(["int"], "int", native)])}, lib)
    count = ns["count"]
    g = ns

    def native_used(expected):
        before = len(lib.calls)
        assert count(10) == expected
        return len(lib.calls) > before

    assert native_used(5)
    g["LIMIT"] = 5  # same type and value: still exact
    assert native_used(5)
    g["LIMIT"] = 5.0  # equal but not the compiled type
    assert not native_used(5)
    g["LIMIT"] = 8
    assert not native_used(8)
    g["LIMIT"] = 5
    assert native_used(5)
    g["step"] = lambda x: x + 100
    assert not native_used(500)
    del g["step"]
    with pytest.raises(NameError):
        count(10)


def test_rebinding_builtins_and_math_falls_back():
    import math

    src = "import math\ndef f(n):\n    t = 0.0\n    for i in range(n):\n        t += abs(i - 2) + math.pi\n    return t\n"
    deps = no_deps(math=["pi"], builtins=["abs", "range"])

    def native(n):
        return sum(abs(i - 2) + math.pi for i in range(n))

    lib = FakeLib()
    ns = accelerate(src, {"f": (deps, [(["int"], "float", native)])}, lib)
    f = ns["f"]
    expected = native(4)
    assert f(4) == expected and len(lib.calls) == 1
    saved_abs, saved_pi = builtins.abs, math.pi
    try:
        builtins.abs = lambda x: 0
        assert f(4) == 4 * math.pi and len(lib.calls) == 1
        builtins.abs = saved_abs
        assert f(4) == expected and len(lib.calls) == 2
        math.pi = 3.0
        assert f(4) == 4 + 4 * 3.0 and len(lib.calls) == 2
        math.pi = saved_pi
        ns["abs"] = lambda x: 1  # a module global shadows the builtin
        assert f(4) == 4 + 4 * math.pi and len(lib.calls) == 2
        del ns["abs"]
        assert f(4) == expected and len(lib.calls) == 3
    finally:
        builtins.abs, math.pi = saved_abs, saved_pi


def test_names_not_defined_yet_raise_like_python():
    src = (
        "def outer(n):\n    t = 0\n    for i in range(n):\n        t += inner(i) + K\n    return t\n"
        "results = []\n"
        "try:\n    outer(3)\nexcept NameError as e:\n    results.append(str(e))\n"
        "def inner(x):\n    return x * 2\n"
        "try:\n    outer(3)\nexcept NameError as e:\n    results.append(str(e))\n"
        "K = 7\nresults.append(outer(3))\n"
    )
    deps = {"funcs": ["inner"], "consts": {"K": ["int", 7]}, "math": None, "builtins": ["range"]}
    lib = FakeLib()
    ns = accelerate(src, {"outer": (deps, [(["int"], "int", lambda n: sum(2 * i + 7 for i in range(n)))])}, lib)
    assert ns["results"] == ["name 'inner' is not defined", "name 'K' is not defined", 27]
    assert len(lib.calls) == 1


def test_one_export_per_signature_dispatches_on_type():
    src = "def fact(n):\n    r = 1\n    while n > 1:\n        r = r * n\n        n = n - 1\n    return r\n"
    lib = FakeLib()

    def f(n):
        r = 1
        while n > 1:
            r, n = r * n, n - 1
        return r

    ns = accelerate(src, {"fact": (no_deps(), [(["int"], "int", f), (["float"], "float", f)])}, lib)
    assert ns["fact"](5) == 120 and ns["fact"](5.0) == 120.0 and ns["fact"](True) == 1
    assert [c[1] for c in lib.calls] == [5, 5.0]
    assert type(lib.calls[1][1]) is float


def test_function_looks_like_the_original():
    filename = "/virtual/prog.py"
    src = (
        "def tri(n, k=2):\n    'Docs.'\n    t = 0\n    for i in range(n):\n        t += 100 // (i - k)\n    return t\n"
        "def call(n):\n    return tri(n)\n"
    )
    lib = FakeLib()
    lib.bail = True  # every call ends up in the Python body
    ns = accelerate(src, {"tri": (no_deps(builtins=["range"]), [(["int", "int"], "int", lambda n, k: 0)])}, lib, filename)
    tri = ns["tri"]
    assert tri.__code__.co_filename == filename and tri.__code__.co_firstlineno == 1
    assert tri.__globals__ is ns and "__pyrust_accel__" not in ns
    assert tri.__name__ == tri.__qualname__ == tri.__code__.co_name == "tri"
    assert tri.__doc__ == "Docs." and tri.__defaults__ == (2,)
    assert not hasattr(tri, "__wrapped__") and tri.__pyrust_native__
    with pytest.raises(ZeroDivisionError) as info:
        ns["call"](5)
    frames = [(f.filename, f.lineno, f.name) for f in traceback.extract_tb(info.value.__traceback__)][1:]
    assert frames == [(filename, 8, "call"), (filename, 5, "tri")]


def test_recursion_through_the_fast_path_costs_one_frame_per_call():
    src = "def fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\n"
    lib = FakeLib()
    lib.bail = True  # like an i64 overflow: Python recursion with a native attempt at each level
    ns = accelerate(src, {"fact": (no_deps(funcs=["fact"]), [(["int"], "int", lambda n: 0)])}, lib)
    limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(400)
        assert len(str(ns["fact"](330))) > 600
    finally:
        sys.setrecursionlimit(limit)


def test_function_rebinding_its_own_name():
    src = "def fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\n"
    lib = FakeLib()
    import math

    ns = accelerate(src, {"fact": (no_deps(funcs=["fact"]), [(["int"], "int", math.factorial)])}, lib)
    f = ns["fact"]
    assert f(10) == 3628800 and len(lib.calls) == 1
    ns["fact"] = lambda n: 0
    assert f(10) == 0 and len(lib.calls) == 1


def test_instrument_drops_the_accel_table_after_the_last_def():
    src = "def f(n):\n    return n\nseen = '__pyrust_accel__' in globals()\ndef g(n):\n    return n\nafter = 1\n"
    ns = {"__pyrust_accel__": {"f": lambda o: o, "g": lambda o: o}}
    exec(instrument(src, "x.py", {"f", "g"}), ns)
    assert ns["seen"] is True and "__pyrust_accel__" not in ns


# --------------------------------------------------------------------------
# Builds (need cargo)
# --------------------------------------------------------------------------


GOOD_AND_BAD = (
    "import collections\n"
    "def good(n):\n    t = 0\n    for i in range(n):\n        t += i\n    return t\n"
    "def bad(n):\n    t = 0\n    for i in range(n):\n        t += i * 2\n    return t\n"
    "def uses_bad(n):\n    t = 0\n    for i in range(n):\n        t += bad(i)\n    return t\n"
    "def also_good(x):\n    t = 0.0\n    for i in range(3):\n        t += x\n    return t\n"
    "print(good(10), bad(10), uses_bad(10), also_good(1.5))\n"
)


@needs_cargo
def test_one_function_failing_to_build_does_not_drop_the_others(tmp_path: Path, monkeypatch):
    import pyrust.codegen as codegen

    real = codegen.generate_rust_module

    def broken(*args, **kwargs):
        rust = real(*args, **kwargs)
        # Simulate a codegen bug confined to `bad`: a type error rustc will reject.
        return rust.replace("pub fn bad(n: i64) -> i64 {", 'pub fn bad(n: i64) -> i64 {\n    let _oops: i64 = "x";', 1)

    monkeypatch.setattr(codegen, "generate_rust_module", broken)
    script = tmp_path / "prog.py"
    script.write_text(GOOD_AND_BAD, encoding="utf-8")
    manifest, plan = prepare_hybrid(script, find_cargo_executable())
    assert manifest is not None
    assert sorted(f["name"] for f in manifest["functions"]) == ["also_good", "good"]
    assert "failed to compile" in manifest["skipped"]["bad"] and "mismatched types" in manifest["skipped"]["bad"]
    assert "bad" in manifest["skipped"]["uses_bad"]
    # `pyrust check` reads the cached manifest and reports each reason.
    checked = plan_execution(script, build=False)
    assert checked.native_functions == ["good", "also_good"]
    assert "failed to compile" in checked.skipped["bad"]


@needs_cargo
def test_multi_signature_build_runs_both_natively(tmp_path: Path):
    script = tmp_path / "prog.py"
    script.write_text(
        "import collections\n"
        "def fact(n):\n    r = 1\n    i = 2\n    while i <= n:\n        r = r * i\n        i += 1\n    return r\n"
        "native = []\n"
        "print(fact(20), fact(20), fact(5.0), fact(30), fact(True))\n",
        encoding="utf-8",
    )
    manifest, _ = prepare_hybrid(script, find_cargo_executable())
    (spec,) = manifest["functions"]
    assert [v["params"] for v in spec["variants"]] == [["int"], ["float"]]
    expected = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True).stdout
    env = dict(os.environ, PYTHONPATH=str(Path(hybrid.__file__).parents[1]))
    got = subprocess.run(
        [sys.executable, "-m", "pyrust.cli", "run", "-q", str(script)], capture_output=True, text=True, env=env
    )
    assert got.stdout == expected, got.stderr
    saved = json.loads((Path(manifest["dll"]).parents[2] / "manifest.json").read_text())
    assert saved["version"] == hybrid.MANIFEST_VERSION


@needs_cargo
def test_exports_are_called_as_builtin_functions(tmp_path: Path):
    from pyrust.hybrid_runner import make_accelerators

    script = tmp_path / "prog.py"
    src = (
        "import collections\n"
        "def total(n, x, flag):\n    t = 0.0\n    for i in range(n):\n        t += x if flag else -x\n    return t\n"
        "def big(n):\n    r = 1\n    for i in range(n):\n        r = r * 3\n    return r\n"
        "def odd(n):\n    r = False\n    for _ in range(1):\n        r = n % 2 == 1\n    return r\n"
        "print(total(3, 1.5, True), big(5), odd(3))\n"
    )
    script.write_text(src, encoding="utf-8")
    manifest, _ = prepare_hybrid(script, find_cargo_executable())
    accel = make_accelerators(manifest)
    ns = {"__name__": "prog", "__pyrust_accel__": accel}
    exec(instrument(src.rsplit("print", 1)[0], str(script), set(accel)), ns)
    for name in ("total", "big", "odd"):
        cells = [c.cell_contents for c in ns[name].__closure__]
        natives = [c for c in cells if type(c).__name__ == "builtin_function_or_method"]
        assert natives and natives[0].__name__ == name
    assert ns["total"](3, 1.5, True) == 4.5 and ns["total"](3, 1.5, False) == -4.5
    assert ns["big"](5) == 243 and ns["big"](50) == 3**50  # overflow: native returns None, Python finishes
    assert ns["odd"](3) is True and ns["odd"](4) is False
