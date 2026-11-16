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


