"""Hybrid tier: native functions inside a CPython program.

When a file cannot be transpiled as a whole (it uses a library or a language
feature PyRust has no Rust equivalent for), it still runs on CPython, but its
self-contained numeric functions are compiled to a native library and swapped
in transparently.

Correctness contract: a native function is only ever a *fast path*.

* Only pure functions are compiled (no I/O, no globals, no side effects), so
  re-running one is always safe.
* The wrapper checks the runtime argument types on every call. Anything that
  does not match a compiled signature goes to the original Python function.
* The wrapper also checks, on every call, that every module global, `math`
  attribute and builtin the native code baked in (transitively, through the
  native functions it calls) is still bound to the object it was compiled
  against. A rebinding (`globals()[...] = ...`, `math.pi = 3`, a star import,
  `builtins.abs = ...`) or a name that is not defined yet sends the call to
  the original Python function, which then behaves exactly like Python.
* If the native code hits anything Python would treat differently (64-bit
  overflow, division by zero, a domain error, an index error, ...) it reports
  failure and the original Python function runs instead, producing the exact
  Python result or exception.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pyrust.errors import PyRustError
from pyrust.types import BOOL, FLOAT, INT

# Builtins with no side effects whose native implementation matches CPython.
PURE_BUILTINS = {
    "len", "range", "abs", "min", "max", "int", "float", "bool", "str",
    "sorted", "list", "set", "dict", "enumerate", "zip", "round", "divmod",
    "all", "any", "reversed", "ord", "chr",
}
if sys.version_info < (3, 12):
    # 3.12+ uses compensated summation for floats; a naive native sum would
    # differ in the last bits.
    PURE_BUILTINS.add("sum")

_SCALAR_NAMES = {INT: "int", FLOAT: "float", BOOL: "bool"}

# At most this many native signatures per function (one per distinct set of
# observed call-site argument types); each extra one costs a Rust module.
MAX_SIGNATURES = 4

# Native builds attempted per program: the first, plus retries that drop the
# function(s) the Rust compiler rejected.
MAX_BUILDS = 4

# Bumped whenever the manifest format changes (older cached manifests are rebuilt).
MANIFEST_VERSION = 2

_BANNED_NODES: tuple[type, ...] = (
    ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await, ast.Try,
    ast.With, ast.AsyncWith, ast.AsyncFor, ast.Lambda, ast.FunctionDef,
    ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Starred,
    ast.LShift, ast.RShift, ast.MatMult, ast.JoinedStr, ast.GeneratorExp,
)
for _name in ("TryStar", "Match"):
    if hasattr(ast, _name):
        _BANNED_NODES += (getattr(ast, _name),)


@dataclass
class Variant:
    """One compiled signature of an accelerated function."""

    params: list[str]
    ret: str
    index: int = 0  # which generated Rust module (accel / accel_v1 / ...) holds it

    def symbol(self, name: str) -> str:
        return f"pyr_export_{name}" if self.index == 0 else f"pyr_export_{name}__v{self.index}"


@dataclass
class AccelFunction:
    name: str
    params: list[str]
    ret: str
    # Every compiled signature; the first one is (params, ret).
    variants: list[Variant] = field(default_factory=list)
    # Names the native code depends on (see `_dependencies`).
    deps: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.variants:
            self.variants = [Variant(list(self.params), self.ret, 0)]


@dataclass
class HybridPlan:
    functions: list[AccelFunction] = field(default_factory=list)
    rust_source: str = ""
    skipped: dict[str, str] = field(default_factory=dict)
    # Rust module per variant index (index 0 is `rust_source`); "" = unused.
    variant_sources: list[str] = field(default_factory=list)
    # Extra facts about accelerated functions (guessed signatures, dropped variants).
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.functions]


# --------------------------------------------------------------------------
# Candidate selection
# --------------------------------------------------------------------------


def _public_names(tree: ast.Module) -> set[str] | None:
    """Names `from <module> import *` binds, from the module's source (None = unknown)."""
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in stmt.targets)
        ):
            if isinstance(stmt.value, (ast.List, ast.Tuple)) and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in stmt.value.elts
            ):
                return {e.value for e in stmt.value.elts}
            return None  # computed __all__: cannot know statically
    return None


def _star_import_names(module: str, level: int, script_dir: Path | None, depth: int = 0) -> set[str] | None:
    """Statically determine the names `from <module> import *` binds (None = unknown).

    Looks for the module the way the program will: next to the script first
    (including `.pyr`), then on `sys.path`. Never imports anything except
    C extension/builtin modules of the standard library.
    """
    import importlib.machinery

    if level:
        return set()  # a script is not a package: the import fails at runtime and binds nothing
    if depth > 3:
        return None
    parts = module.split(".")
    source_path: Path | None = None
    if script_dir is not None:
        base = script_dir.joinpath(*parts)
        for cand in (base.with_suffix(".py"), base.with_suffix(".pyr"), base / "__init__.py", base / "__init__.pyr"):
            if cand.is_file():
                source_path = cand
                break
    if source_path is None:
        try:
            spec = importlib.machinery.PathFinder.find_spec(parts[0])
            for part in parts[1:]:
                if spec is None or not spec.submodule_search_locations:
                    spec = None
                    break
                spec = importlib.machinery.PathFinder.find_spec(part, list(spec.submodule_search_locations))
        except (ImportError, ValueError, OSError):
            spec = None
        if spec is None and module in sys.builtin_module_names:
            spec = importlib.machinery.ModuleSpec(module, None, origin="built-in")
        if spec is None:
            return None
        origin = spec.origin or ""
        if origin.endswith((".py", ".pyr")):
            source_path = Path(origin)
        elif parts[0] in getattr(sys, "stdlib_module_names", ()) and (
            origin == "built-in" or origin.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
        ):
            try:  # a compiled stdlib module: importing it runs no Python code
                import importlib

                mod = importlib.import_module(module)
            except Exception:
                return None
            names = getattr(mod, "__all__", None)
            if names is None:
                names = [n for n in dir(mod) if not n.startswith("_")]
            return set(names)
        else:
            return None
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None
    explicit = _public_names(tree)
    if explicit is not None:
        return explicit
    if any(
        isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in s.targets)
        for s in tree.body
    ):
        return None
    counts, unknown = _module_bindings(tree, source_path.parent, depth + 1, package=source_path.name.startswith("__init__"))
    if unknown:
        return None
    return {n for n in counts if not n.startswith("_")}


def _module_bindings(
    tree: ast.Module, script_dir: Path | None = None, depth: int = 0, package: bool = False
) -> tuple[dict[str, int], list[tuple[str, int]]]:
    """How many times each name is bound at module level (or via `global`).

    Also returns the star imports whose names cannot be determined statically,
    as (module, line) pairs: any global may be rebound by those.
    """
    counts: dict[str, int] = {}
    unknown: list[tuple[str, int]] = []

    def bump(name: str, n: int = 1) -> None:
        counts[name] = counts.get(name, 0) + n

    def visit(node: ast.AST, top: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if top:
                bump(node.name)
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Global, ast.Nonlocal)):
                    for n in sub.names:
                        bump(n, 2)  # rebindable from elsewhere: never "bound once"
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bump(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bump(node.name, 2)  # `except E as NAME` binds NAME and then unbinds it
        elif hasattr(ast, "MatchAs") and isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bump(node.name)  # `case NAME:` / `case [*NAME]` capture patterns
        elif hasattr(ast, "MatchMapping") and isinstance(node, ast.MatchMapping) and node.rest:
            bump(node.rest)  # `case {**NAME}`
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bump((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    bump(alias.asname or alias.name)
                    continue
                if package and node.level:
                    names = None  # relative star import inside a package: do not chase
                else:
                    names = _star_import_names(node.module or "", node.level, script_dir, depth)
                if names is None:
                    unknown.append((("." * node.level) + (node.module or ""), node.lineno))
                else:
                    for n in names:
                        bump(n)
        for child in ast.iter_child_nodes(node):
            visit(child, top)

    for stmt in tree.body:
        visit(stmt, True)
    return counts, unknown


def _math_alias_ok(tree: ast.Module, bindings: dict[str, int]) -> bool:
    if bindings.get("math", 0) != 1:
        return False
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "math" and alias.asname is None:
                    return True
    return False


def _module_consts(tree: ast.Module, bindings: dict[str, int]) -> dict[str, ast.Assign]:
    consts: dict[str, ast.Assign] = {}
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and type(stmt.value.value) in (int, float, bool)
            and bindings.get(stmt.targets[0].id) == 1
        ):
            if type(stmt.value.value) is int and not (-(2**63) <= stmt.value.value < 2**63):
                continue
            consts[stmt.targets[0].id] = stmt
    return consts


def _static_reason(fn: ast.FunctionDef, bindings: dict[str, int]) -> str | None:
    if fn.decorator_list:
        return "is decorated"
    a = fn.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
        return "uses *args/**kwargs/keyword-only/positional-only parameters"
    if any(not isinstance(d, ast.Constant) for d in a.defaults):
        return "has non-literal default arguments"
    if fn.name == "main":
        return "is the program entry point"
    if bindings.get(fn.name, 0) != 1:
        return "is rebound elsewhere in the module"
    if any(n.startswith("__pyr_") for n in _all_names(fn)):
        return "uses a name reserved by PyRust (__pyr_...)"
    return None


def _all_names(fn: ast.FunctionDef) -> set[str]:
    out = {a.arg for a in fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs}
    out |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    return out


def _local_names(fn: ast.FunctionDef) -> set[str]:
    names = {x.arg for x in fn.args.posonlyargs + fn.args.args}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
    return names


def _body_reason(
    fn: ast.FunctionDef,
    candidates: set[str],
    bindings: dict[str, int],
    consts: dict[str, ast.Assign],
    math_ok: bool,
) -> str | None:
    """Why `fn` cannot be compiled as a pure native function (None = it can)."""
    local_names = _local_names(fn)

    call_funcs: set[int] = set()
    attr_bases: set[int] = set()
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if isinstance(node, _BANNED_NODES):
                return f"uses {type(node).__name__} (line {getattr(node, 'lineno', fn.lineno)})"
            if isinstance(node, ast.Call):
                if node.keywords:
                    return f"passes keyword arguments (line {node.lineno})"
                f = node.func
                if isinstance(f, ast.Name):
                    call_funcs.add(id(f))
                    if f.id in local_names:
                        return f"calls a local value '{f.id}'"
                    if f.id in candidates:
                        continue
                    if f.id in bindings:
                        return f"calls '{f.id}', which stays on CPython"
                    if f.id not in PURE_BUILTINS:
                        return f"calls '{f.id}', which is not a pure builtin"
                elif isinstance(f, ast.Attribute):
                    base = f.value
                    if isinstance(base, ast.Name) and base.id not in local_names:
                        if not (base.id == "math" and math_ok):
                            return f"calls '{base.id}.{f.attr}', which stays on CPython"
                else:
                    return f"calls a computed function (line {node.lineno})"
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                attr_bases.add(id(node.value))

    for stmt in fn.body:
        for node in ast.walk(stmt):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                continue
            if node.id in local_names or id(node) in call_funcs:
                continue
            if node.id == "math" and math_ok and id(node) in attr_bases:
                continue
            if node.id in consts:
                continue
            return f"reads global '{node.id}'"
    return None


_LOOPS = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp)


def _native_calls(candidates: dict[str, ast.FunctionDef]) -> dict[str, set[str]]:
    return {
        name: {
            n.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in candidates
        }
        for name, fn in candidates.items()
    }


def _reach(calls: dict[str, set[str]], start: str) -> set[str]:
    """Every function `start` calls natively, directly or not (excluding itself unless recursive)."""
    seen: set[str] = set()
    stack = list(calls.get(start, ()))
    while stack:
        cur = stack.pop()
        if cur not in seen:
            seen.add(cur)
            stack.extend(calls.get(cur, ()))
    return seen


def _worth_exporting(candidates: dict[str, ast.FunctionDef]) -> set[str]:
    """Functions with enough work inside to repay the Python<->native bridge.

    Crossing the bridge costs more than a straight-line function saves, so only
    functions that loop, recurse, or call one that does are swapped in. The
    others are still compiled and used by native callers.
    """
    calls = _native_calls(candidates)
    reach = {name: _reach(calls, name) for name in candidates}
    heavy = {
        name
        for name, fn in candidates.items()
        if name in reach[name] or any(isinstance(n, _LOOPS) for n in ast.walk(fn))
    }
    return {name for name in candidates if name in heavy or reach[name] & heavy}


def _dependencies(
    name: str,
    candidates: dict[str, ast.FunctionDef],
    calls: dict[str, set[str]],
    consts: dict[str, ast.Assign],
    math_ok: bool,
) -> dict:
    """Every global the native code of `name` baked in, transitively.

    * funcs: module functions called natively (their native code is used, so
      the global must still be that function);
    * consts: module constants inlined into the native code, with their value;
    * math: `math` attributes used (None if `math` is not used at all);
    * builtins: builtins called.
    """
    members = {name} | _reach(calls, name)
    funcs: set[str] = set()
    const_names: set[str] = set()
    builtins_used: set[str] = set()
    math_attrs: set[str] = set()
    uses_math = False
    for member in members:
        fn = candidates[member]
        local = _local_names(fn)
        funcs |= calls.get(member, set())
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "math" and math_ok and "math" not in local:
                    uses_math = True
                    math_attrs.add(node.attr)
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)) or node.id in local:
                continue
            if node.id in consts:
                const_names.add(node.id)
            elif node.id in PURE_BUILTINS and node.id not in candidates:
                builtins_used.add(node.id)
    const_values: dict[str, list] = {}
    for c in sorted(const_names):
        value = consts[c].value.value
        if type(value) is float:
            const_values[c] = ["float", value.hex()]
        elif type(value) is bool:
            const_values[c] = ["bool", value]
        else:
            const_values[c] = ["int", value]
    return {
        "funcs": sorted(funcs),
        "consts": const_values,
        "math": sorted(math_attrs) if uses_math else None,
        "builtins": sorted(builtins_used),
    }


# --------------------------------------------------------------------------
# Call-site type hints (best effort: the runtime guard makes guesses safe)
# --------------------------------------------------------------------------


class _Elems:
    """A loop variable bound from the elements of a literal: its kind is their common kind."""

    def __init__(self, nodes: list[ast.AST]) -> None:
        self.nodes = nodes


def _literal_elements(it: ast.AST, target: ast.AST) -> dict[str, _Elems | str]:
    """Kinds of loop variables iterating over a list/tuple/set literal."""
    out: dict[str, _Elems | str] = {}
    if not isinstance(it, (ast.List, ast.Tuple, ast.Set)) or not it.elts:
        return out
    if isinstance(target, ast.Name):
        out[target.id] = _Elems(list(it.elts))
    elif isinstance(target, (ast.Tuple, ast.List)) and all(isinstance(t, ast.Name) for t in target.elts):
        width = len(target.elts)
        rows = it.elts
        if all(isinstance(r, (ast.Tuple, ast.List)) and len(r.elts) == width for r in rows):
            for i, t in enumerate(target.elts):
                out[t.id] = _Elems([r.elts[i] for r in rows])
        else:
            for t in target.elts:
                out[t.id] = "?"
    return out


class _Hints:
    def __init__(self, tree: ast.Module, candidates: set[str]) -> None:
        self.candidates = candidates
        self._counts: dict[str, dict[tuple[str, ...], int]] = {}
        self._scan_scope(tree.body, skip_candidates=True)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in candidates and node in tree.body:
                    continue
                self._scan_scope(node.body, skip_candidates=False)
        # Most frequent signature first (ties: first seen), at most MAX_SIGNATURES.
        self.hints: dict[str, list[tuple[str, ...]]] = {
            name: sorted(c, key=lambda s, c=c: -c[s])[:MAX_SIGNATURES] for name, c in self._counts.items()
        }

    def _scan_scope(self, body: list[ast.stmt], skip_candidates: bool) -> None:
        nodes: list[ast.AST] = []
        stack: list[ast.AST] = list(reversed(body))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            nodes.append(node)
            stack.extend(reversed(list(ast.iter_child_nodes(node))))

        stores: dict[str, list[ast.AST | str | _Elems]] = {}
        handled: set[int] = set()
        for node in nodes:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                stores.setdefault(node.targets[0].id, []).append(node.value)
                handled.add(id(node.targets[0]))
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                stores.setdefault(node.target.id, []).append(
                    ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()), op=node.op, right=node.value)
                )
                handled.add(id(node.target))
            elif isinstance(node, ast.For):
                it = node.iter
                literal = _literal_elements(it, node.target)
                if isinstance(node.target, ast.Name):
                    is_range = isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
                    stores.setdefault(node.target.id, []).append(
                        "int" if is_range else literal.get(node.target.id, "?")
                    )
                    handled.add(id(node.target))
                elif literal:
                    for t in node.target.elts:
                        stores.setdefault(t.id, []).append(literal[t.id])
                        handled.add(id(t))
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and id(node) not in handled:
                stores.setdefault(node.id, []).append("?")

        env: dict[str, str] = {}
        for _ in range(3):
            for name, values in stores.items():
                kinds = {self._kind_of(v, env, name) for v in values}
                kinds.discard("self")
                if len(kinds) == 1 and "?" not in kinds:
                    env[name] = next(iter(kinds))
                else:
                    env.pop(name, None)

        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.candidates:
                if node.keywords:
                    continue
                kinds = [self._simple(a, env, None) for a in node.args]
                if all(k in ("int", "float", "bool") for k in kinds):
                    sig = tuple(k for k in kinds if k)
                    per_fn = self._counts.setdefault(node.func.id, {})
                    per_fn[sig] = per_fn.get(sig, 0) + 1

    def _kind_of(self, v: ast.AST | str | _Elems, env: dict[str, str], name: str) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, _Elems):
            kinds = {self._simple(e, env, None) for e in v.nodes}
            return next(iter(kinds)) if len(kinds) == 1 and None not in kinds else "?"
        return self._simple(v, env, name) or "?"

    def _simple(self, node: ast.AST, env: dict[str, str], self_name: str | None) -> str | None:
        if isinstance(node, ast.Constant):
            return {bool: "bool", int: "int", float: "float"}.get(type(node.value))
        if isinstance(node, ast.Name):
            if node.id == self_name:
                return env.get(node.id, "self")
            return env.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            k = self._simple(node.operand, env, self_name)
            return "int" if k == "bool" else k
        if isinstance(node, ast.BinOp):
            lk = self._simple(node.left, env, self_name)
            rk = self._simple(node.right, env, self_name)
            if lk == "self":
                lk = rk
            if rk == "self":
                rk = lk
            if lk is None or rk is None or lk == "self":
                return None
            if isinstance(node.op, ast.Div):
                return "float"
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
                return "float" if "float" in (lk, rk) else "int"
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id in ("len", "int", "ord") or (node.func.id == "round" and len(node.args) == 1):
                return "int"
            if node.func.id == "float":
                return "float"
        return None


_HINT_LITERAL = {"int": "0", "float": "0.0", "bool": "True"}


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def _unparse_fn(fn: ast.FunctionDef) -> str:
    return ast.unparse(fn)


@dataclass
class _Compiled:
    env: object
    rust: str
    guessed: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]  # name -> (int guess, float guess)


def _compile(
    candidates: dict[str, ast.FunctionDef],
    sigs: dict[str, list[tuple[str, ...]]],
    consts: dict[str, ast.Assign],
    math_ok: bool,
    filename: str,
    prune,
    drop,
    note,
) -> _Compiled | None:
    """Infer types for `candidates` under the call signatures `sigs` and generate their Rust.

    `sigs[name][0]` is the signature compiled for `name` (absent: inferred from
    native callers, else guessed). A function the compiler rejects is retried
    with its next signature, then dropped (`drop(name, reason)`), and `prune()`
    removes whatever depended on it. Mutates `candidates` and `sigs`.
    """
    from pyrust.codegen import generate_rust_module
    from pyrust.inference import _positional_params, _TypeInferer
    from pyrust.ownership import analyze_ownership
    from pyrust.parser import parse_source
    from pyrust.types import ANY, UNKNOWN

    int_guess: dict[str, tuple[str, ...]] = {}
    float_guess: dict[str, tuple[str, ...]] = {}
    budget = len(candidates) * 2 + 4 + sum(len(s) for s in sigs.values())
    for _attempt in range(budget):
        if not candidates:
            return None
        lines: list[str] = []
        spans: list[tuple[int, int, str]] = []
        if math_ok:
            lines.append("import math")
        used_consts = [n for n in consts if any(
            isinstance(x, ast.Name) and x.id == n for fn in candidates.values() for x in ast.walk(fn)
        )]
        for n in used_consts:
            lines.append(ast.unparse(consts[n]))
        for name, fn in candidates.items():
            text = _unparse_fn(fn).splitlines()
            spans.append((len(lines) + 1, len(lines) + len(text), name))
            lines.extend(text)
        for name in candidates:
            sig = sigs[name][0] if sigs.get(name) else int_guess.get(name)
            if sig is not None:
                lines.append(f"{name}({', '.join(_HINT_LITERAL[k] for k in sig)})")
        synth = "\n".join(lines) + "\n"

        try:
            parsed = parse_source(synth, filename=filename, module_name="accel")
            inferer = _TypeInferer(parsed)
            env = inferer.run()

            # Parameters nothing pinned down: guess, deliberately. The runtime
            # guard sends every other argument type to the original function,
            # so a wrong guess only costs speed, never correctness.
            guessed = False
            for name, fn in candidates.items():
                if name in int_guess or sigs.get(name):
                    continue
                ft = env.func_types[name]
                n_pos = len(_positional_params(fn))
                if any(p in (UNKNOWN, ANY) for p in ft.params[:n_pos]):
                    int_guess[name] = tuple(
                        "int" if p in (UNKNOWN, ANY) else _SCALAR_NAMES.get(p, "int") for p in ft.params[:n_pos]
                    )
                    float_guess[name] = tuple(
                        "float" if p in (UNKNOWN, ANY) else _SCALAR_NAMES.get(p, "int") for p in ft.params[:n_pos]
                    )
                    guessed = True
            if guessed:
                continue

            inferer._finalize_unknown_params()
            ownership = analyze_ownership(parsed, env)
            rust = generate_rust_module(parsed, env, ownership, is_entry=False, all_modules=["accel"])
        except PyRustError as exc:
            culprit = next((n for lo, hi, n in spans if exc.lineno and lo <= exc.lineno <= hi), None)
            if culprit is None:
                for name in list(candidates):
                    drop(name, f"native compile failed: {exc.message}")
                candidates.clear()
                return None
            if len(sigs.get(culprit, ())) > 1:
                note(culprit, f"({', '.join(sigs[culprit][0])}) not compiled: {exc.message}")
                sigs[culprit].pop(0)
                continue
            drop(culprit, f"not supported natively yet: {exc.message}")
            del candidates[culprit]
            prune()
            continue
        except Exception as exc:  # inference/codegen bug: never take the program down
            for name in list(candidates):
                drop(name, f"native compile failed: {type(exc).__name__}: {exc}")
            candidates.clear()
            return None
        guesses = {n: (int_guess[n], float_guess[n]) for n in int_guess if n in candidates}
        return _Compiled(env, rust, guesses)
    return None


def _int_only_params(fn: ast.FunctionDef) -> set[str]:
    """Names used where only an int works (range() bounds, indices, chr())."""
    out: set[str] = set()
    for node in ast.walk(fn):
        spots: list[ast.AST] = []
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("range", "chr"):
            spots = list(node.args)
        elif isinstance(node, ast.Subscript):
            spots = [node.slice]
        for spot in spots:
            out |= {n.id for n in ast.walk(spot) if isinstance(n, ast.Name)}
    return out


def _scalar_signature(env, name: str, fn: ast.FunctionDef) -> tuple[list[str], str] | None:
    from pyrust.inference import _positional_params

    ft = env.func_types[name]
    n_pos = len(_positional_params(fn))
    params = [_SCALAR_NAMES.get(p) for p in ft.params[:n_pos]]
    ret = _SCALAR_NAMES.get(ft.ret)
    if ret is None or any(p is None for p in params) or len(ft.params) != n_pos:
        return None
    return [p for p in params if p], ret


def plan_hybrid(
    source: str,
    filename: str = "<string>",
    exclude: dict[tuple[int, str], str] | None = None,
) -> HybridPlan:
    """Work out which top-level functions can run natively, and generate their Rust.

    `exclude` maps (variant index, function name) to the reason it must not be
    compiled there (used to retry after the Rust compiler rejected it).
    """
    plan = HybridPlan()
    exclude = dict(exclude or {})
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return plan

    script_dir = Path(filename).resolve().parent if filename and not filename.startswith("<") else None
    bindings, unknown_stars = _module_bindings(tree, script_dir)
    consts = _module_consts(tree, bindings)
    math_ok = _math_alias_ok(tree, bindings)
    all_fns = {s.name: s for s in tree.body if isinstance(s, ast.FunctionDef)}

    candidates: dict[str, ast.FunctionDef] = {}
    for name, fn in all_fns.items():
        reason = _static_reason(fn, bindings)
        if reason is None and unknown_stars:
            mod, line = unknown_stars[0]
            reason = (
                f"'from {mod} import *' (line {line}) can rebind any global it uses, "
                "and its names cannot be determined statically"
            )
        if reason is None and (0, name) in exclude:
            reason = exclude[(0, name)]
        if reason:
            plan.skipped[name] = reason
        else:
            candidates[name] = fn

    def make_prune(cands: dict[str, ast.FunctionDef], drop):
        def prune() -> None:
            changed = True
            while changed:
                changed = False
                for name in list(cands):
                    reason = _body_reason(cands[name], set(cands), bindings, consts, math_ok)
                    if reason:
                        drop(name, reason)
                        del cands[name]
                        changed = True
        return prune

    notes: dict[str, list[str]] = {}

    def note(name: str, text: str) -> None:
        notes.setdefault(name, []).append(text)

    def drop0(name: str, reason: str) -> None:
        plan.skipped[name] = reason

    prune0 = make_prune(candidates, drop0)
    prune0()
    if not candidates:
        return plan

    # Variant 0: every function with its most common observed signature.
    hints = _Hints(tree, set(candidates)).hints
    sigs0 = {name: list(hints.get(name, [])) for name in candidates}
    compiled = _compile(candidates, sigs0, consts, math_ok, filename, prune0, drop0, note)
    if compiled is None:
        return plan

    worth = _worth_exporting(candidates)
    calls = _native_calls(candidates)
    exported: dict[str, AccelFunction] = {}
    extra: dict[str, list[tuple[str, ...]]] = {}
    for name, fn in candidates.items():
        if name not in worth:
            plan.skipped[name] = (
                "too small to benefit (no loop or recursion); calling into native code would cost more "
                "than it saves"
            )
            continue
        sig = _scalar_signature(compiled.env, name, fn)
        if sig is None:
            plan.skipped[name] = "compiled natively as a helper only (arguments/result are not plain int/float/bool)"
            continue
        params, ret = sig
        exported[name] = AccelFunction(
            name, params, ret, deps=_dependencies(name, candidates, calls, consts, math_ok)
        )
        if sigs0.get(name):
            extra[name] = sigs0[name][1:]
        elif name in compiled.guessed:
            int_sig, float_sig = compiled.guessed[name]
            int_only = _int_only_params(fn)  # range() bounds, indices: a float would only raise
            float_sig = tuple(
                i if a.arg in int_only else f for a, i, f in zip(fn.args.args, int_sig, float_sig)
            )
            note(
                name,
                f"no call site pins the argument types: compiled for guessed ({', '.join(params)})"
                + (f" and ({', '.join(float_sig)})" if float_sig != int_sig else "")
                + " arguments; other argument types run on CPython",
            )
            if float_sig != int_sig:
                extra[name] = [float_sig]

    plan.functions = list(exported.values())
    plan.rust_source = compiled.rust if plan.functions else ""
    plan.variant_sources = [plan.rust_source]

    # Variants 1..: one more native signature per function whose call sites disagree.
    for index in range(1, MAX_SIGNATURES):
        targets = {
            name: [extra[name][index - 1]]
            for name in extra
            if len(extra[name]) >= index and (index, name) not in exclude
        }
        for name in extra:
            if len(extra[name]) >= index and (index, name) in exclude:
                note(name, f"({', '.join(extra[name][index - 1])}) dropped: {exclude[(index, name)]}")
        if not targets:
            break
        members: set[str] = set(targets)
        for name in targets:
            members |= _reach(calls, name)
        subset = {n: candidates[n] for n in candidates if n in members and (index, n) not in exclude}

        def drop_v(name: str, reason: str, _targets=targets) -> None:
            if name in _targets:
                note(name, f"({', '.join(_targets[name][0])}) not compiled: {reason}")

        result = _compile(subset, targets, consts, math_ok, filename, make_prune(subset, drop_v), drop_v, note)
        added = False
        if result is not None:
            for name in targets:
                if name not in subset:
                    continue
                sig = _scalar_signature(result.env, name, subset[name])
                fn_plan = exported[name]
                if sig is None:
                    note(name, f"({', '.join(targets[name][0])}) not compiled: arguments/result are not plain scalars")
                    continue
                if any(v.params == sig[0] for v in fn_plan.variants):
                    continue
                fn_plan.variants.append(Variant(sig[0], sig[1], index))
                added = True
        plan.variant_sources.append(result.rust if (result is not None and added) else "")

    plan.notes = {name: "; ".join(texts) for name, texts in notes.items() if name in exported}
    return plan


# --------------------------------------------------------------------------
# Crate generation and build
# --------------------------------------------------------------------------

_RUST_ARG = {"int": "i64", "float": "f64", "bool": "u8"}
_RUST_OUT = {"int": "i64", "float": "f64", "bool": "u8"}


_RUST_FAIL = {"int": "i64::MIN", "float": "f64::NAN", "bool": "2"}


def _module_name(index: int) -> str:
    return "accel" if index == 0 else f"accel_v{index}"


def _lib_rs(plan: HybridPlan) -> str:
    out = [
        "#![allow(dead_code, unused_imports, unused_variables, unused_mut, non_snake_case)]",
        "mod pyrust_rt;",
    ]
    for index, src in enumerate(plan.variant_sources or [plan.rust_source]):
        if src:
            out.append(f"mod {_module_name(index)};")
    out += [
        "",
        "thread_local! {",
        "    static PYR_FAILED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };",
        "}",
        "",
        "#[no_mangle]",
        'pub extern "C" fn pyr_init() {',
        "    // Failures are reported to the Python wrapper, which re-runs the",
        "    // original function; keep stderr clean.",
        "    std::panic::set_hook(Box::new(|_| {}));",
        "}",
        "",
        "/// 1 if the last export on this thread bailed out (and clears the flag).",
        "#[no_mangle]",
        'pub extern "C" fn pyr_take_error() -> i32 {',
        "    PYR_FAILED.with(|f| f.replace(false)) as i32",
        "}",
        "",
    ]
    for fn in plan.functions:
        for var in fn.variants:
            params = [f"a{i}: {_RUST_ARG[t]}" for i, t in enumerate(var.params)]
            args = [f"a{i} != 0" if t == "bool" else f"a{i}" for i, t in enumerate(var.params)]
            conv = "v as u8" if var.ret == "bool" else "v"
            out += [
                "#[no_mangle]",
                f'pub extern "C" fn {var.symbol(fn.name)}({", ".join(params)}) -> {_RUST_OUT[var.ret]} {{',
                f"    match std::panic::catch_unwind(|| {_module_name(var.index)}::{fn.name}({', '.join(args)})) {{",
                f"        Ok(v) => {conv},",
                "        Err(_) => {",
                "            PYR_FAILED.with(|f| f.set(true));",
                f"            {_RUST_FAIL[var.ret]}",
                "        }",
                "    }",
                "}",
                "",
            ]
    return "\n".join(out)


_CARGO_TOML = """[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[profile.release]
opt-level = 3
lto = "thin"
overflow-checks = true
panic = "unwind"
"""


def _dylib_name(crate: str) -> str:
    if os.name == "nt":
        return f"{crate}.dll"
    if sys.platform == "darwin":
        return f"lib{crate}.dylib"
    return f"lib{crate}.so"


from pyrust.plancache import toolchain_salt  # noqa: E402  (re-exported)


_RUST_ERROR = re.compile(r"^(src[\\/][\w./\\]+\.rs):(\d+):\d+: error(?:\[\w+\])?: (.*)$", re.M)
_RUST_FN = re.compile(r"^(?:pub )?fn (\w+)\s*[<(]")
_RUST_EXPORT = re.compile(r'^pub extern "C" fn (\w+)\(')


def _build_culprits(stderr: str, plan: HybridPlan) -> dict[tuple[int, str], str]:
    """Map rustc errors back to the (variant, function) whose generated code they are in."""
    owners: dict[str, list[tuple[int, str]]] = {}  # file -> [(first line, function)]
    for index, src in enumerate(plan.variant_sources or [plan.rust_source]):
        if not src:
            continue
        starts = []
        for lineno, line in enumerate(src.splitlines(), 1):
            m = _RUST_FN.match(line)
            if m:
                starts.append((lineno, m.group(1)))
        owners[f"src/{_module_name(index)}.rs"] = starts
    symbols = {var.symbol(fn.name): (var.index, fn.name) for fn in plan.functions for var in fn.variants}
    lib_starts = []
    for lineno, line in enumerate(_lib_rs(plan).splitlines(), 1):
        m = _RUST_EXPORT.match(line)
        if m and m.group(1) in symbols:
            lib_starts.append((lineno, m.group(1)))

    found: dict[tuple[int, str], str] = {}
    for m in _RUST_ERROR.finditer(stderr):
        path, lineno, msg = m.group(1).replace("\\", "/"), int(m.group(2)), m.group(3).strip()
        if path == "src/lib.rs":
            owner = [s for start, s in lib_starts if start <= lineno]
            if owner:
                found.setdefault(symbols[owner[-1]], msg)
            continue
        if path not in owners:
            continue
        index = 0 if path == "src/accel.rs" else int(path.rsplit("_v", 1)[1].split(".")[0])
        owner = [name for start, name in owners[path] if start <= lineno]
        if owner:
            found.setdefault((index, owner[-1]), msg)
    return found


def _write_crate(crate_dir: Path, crate: str, plan: HybridPlan, runtime: str) -> None:
    src = crate_dir / "src"
    src.mkdir(parents=True, exist_ok=True)
    (crate_dir / "Cargo.toml").write_text(_CARGO_TOML.format(name=crate), encoding="utf-8")
    (src / "pyrust_rt.rs").write_text(runtime, encoding="utf-8")
    for old in src.glob("accel*.rs"):
        old.unlink()
    for index, text in enumerate(plan.variant_sources or [plan.rust_source]):
        if text:
            (src / f"{_module_name(index)}.rs").write_text(text, encoding="utf-8")
    (src / "lib.rs").write_text(_lib_rs(plan), encoding="utf-8")


def _manifest_functions(plan: HybridPlan) -> list[dict]:
    return [
        {
            "name": f.name,
            "params": f.params,
            "ret": f.ret,
            "variants": [{"params": v.params, "ret": v.ret, "symbol": v.symbol(f.name)} for v in f.variants],
            "deps": f.deps,
        }
        for f in plan.functions
    ]


def prepare_hybrid(script: Path, cargo_bin: str | None) -> tuple[dict | None, HybridPlan | None]:
    """Plan + build (cached) the native library for `script`.

    Returns `(manifest, plan)`; manifest is None when nothing runs natively.
    `plan` is None when the result came from the cache.

    If the Rust compiler rejects some function's generated code, that function
    (and whatever calls it natively) is dropped and the rest is rebuilt, at most
    MAX_BUILDS builds in all; each dropped function's reason is recorded.
    """
    from pyrust.runtime_template import RUNTIME_RS

    script = Path(script).resolve()
    source = script.read_text(encoding="utf-8")
    digest = hashlib.sha256((toolchain_salt() + "\0" + source).encode("utf-8")).hexdigest()[:16]
    stem = "".join(c if c.isalnum() else "_" for c in script.stem.lower()) or "app"
    crate = f"pyr_{stem}_accel"
    crate_dir = script.parent / ".pyrust_cache" / f"{stem}_hybrid_{digest}"
    manifest_path = crate_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") == MANIFEST_VERSION and (
                not manifest["functions"] or Path(manifest["dll"]).exists()
            ):
                return (manifest if manifest["functions"] else None), None
        except (OSError, ValueError, KeyError):
            pass

    exclude: dict[tuple[int, str], str] = {}
    plan = plan_hybrid(source, str(script))
    manifest: dict = {"version": MANIFEST_VERSION, "dll": "", "functions": [], "skipped": plan.skipped}
    if plan.functions and cargo_bin:
        for attempt in range(MAX_BUILDS):
            _write_crate(crate_dir, crate, plan, RUNTIME_RS)
            proc = subprocess.run(
                [cargo_bin, "build", "--quiet", "--release", "--message-format=short"],
                cwd=str(crate_dir), capture_output=True, text=True, check=False,
            )
            dll = crate_dir / "target" / "release" / _dylib_name(crate)
            if proc.returncode == 0 and dll.exists():
                manifest["dll"] = str(dll)
                manifest["functions"] = _manifest_functions(plan)
                break
            culprits = _build_culprits(proc.stderr, plan)
            if not culprits or attempt == MAX_BUILDS - 1:
                manifest["error"] = proc.stderr.strip()[-4000:]
                for f in plan.functions:
                    plan.skipped[f.name] = "generated Rust failed to compile (PyRust bug); running on CPython"
                plan.functions = []
                break
            for key, msg in culprits.items():
                exclude[key] = f"generated Rust failed to compile (PyRust bug: {msg}); running on CPython"
            plan = plan_hybrid(source, str(script), exclude=exclude)
            manifest["skipped"] = plan.skipped
            if not plan.functions:
                break
        manifest["skipped"] = plan.skipped
        if plan.notes and manifest["functions"]:
            manifest["notes"] = plan.notes
    elif plan.functions:
        return None, plan  # no cargo: do not cache, it may be installed later
    try:
        crate_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError:
        pass
    return (manifest if manifest["functions"] else None), plan
