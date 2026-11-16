"""Runs a `.pyr`/`.py` program on CPython, optionally with native functions swapped in.

Used in-process by the `pyrust` command and in a fresh interpreter by the
`run_project` API. Imports nothing from the rest of PyRust, and loads `ctypes`
and `ast` only when native functions are actually present, so a program with
nothing to accelerate starts about as fast as plain `python script.py`.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

_GUARD = {
    # Closure cells, not builtins: a parameter or global named `type`/`int`/... must not matter.
    "int": "__pyr_type({a}) is __pyr_int and -9223372036854775808 <= {a} <= 9223372036854775807",
    "float": "__pyr_type({a}) is __pyr_float",
    "bool": "__pyr_type({a}) is __pyr_bool",
}

# How each return type tells "native result" from "native code bailed out".
# The native side returns a sentinel on failure and raises a per-thread flag,
# so the common case costs one comparison and no extra foreign call.
_RETURN = {
    "int": "if __pyr_r != -9223372036854775808 or not __pyr_err():\n    return __pyr_r\n",
    "float": "if __pyr_r == __pyr_r or not __pyr_err():\n    return __pyr_r\n",
    "bool": "if __pyr_r < 2:\n    return __pyr_r == 1\n",
    # Exports called as CPython builtin functions return the result object, or None on failure.
    "object": "if __pyr_r is not None:\n    return __pyr_r\n",
}

# The `def` statement of each accelerated function, by (filename, name), saved
# by `instrument` so `_wrap` can compile the fast path *into* the function: no
# extra frame (tracebacks and recursion depth match Python), and the code keeps
# its real filename, line numbers and globals.
_DEFS: dict = {}
_UNSET = object()


class _Context:
    """State shared by the accelerated functions of one native library."""

    def __init__(self, take_error) -> None:
        import builtins

        self.take_error = take_error
        # What the native code was compiled against, captured before the program runs.
        self.builtins = dict(builtins.__dict__)
        self._math = None
        self.cells: dict = {}

    @property
    def math(self):
        """The standard `math` module the program uses, or _UNSET if it is shadowed.

        Not imported up front: a program's own `import math` must find a local
        `math.py` first, exactly as under plain Python.
        """
        if self._math is None:
            m = sys.modules.get("math")
            if m is None:
                import math as m
            origin = getattr(getattr(m, "__spec__", None), "origin", None) or ""
            genuine = isinstance(m, types.ModuleType) and not origin.endswith((".py", ".pyr", ".pyc"))
            self._math = m if genuine else _UNSET
        return self._math

    def math_attr(self, attr: str):
        """`math.<attr>` if it is still the standard object (else _UNSET: never take the fast path)."""
        m = self.math
        if m is _UNSET:
            return _UNSET
        v = m.__dict__.get(attr, _UNSET)
        consts = {"pi": 3.141592653589793, "e": 2.718281828459045, "tau": 6.283185307179586, "inf": float("inf")}
        if attr in consts:
            return v if type(v) is float and v == consts[attr] else _UNSET
        if attr == "nan":
            return v if type(v) is float and v != v else _UNSET
        if type(v) is types.BuiltinFunctionType and getattr(v, "__self__", None) is m and v.__name__ == attr:
            return v
        return _UNSET

    def cell(self, kind: str, name: str):
        """A closure cell shared by every fast path: the object a global must still be bound to."""
        key = (kind, name)
        c = self.cells.get(key)
        if c is None:
            c = self.cells[key] = types.CellType(_UNSET)
        return c

    def register(self, name: str, fn):
        """Record the function object (and its code and defaults) native callers stand for."""
        self.cell("fn", name).cell_contents = fn
        self.cell("code", name).cell_contents = getattr(fn, "__code__", _UNSET)
        self.cell("defaults", name).cell_contents = getattr(fn, "__defaults__", _UNSET)
        return fn

    def refresher(self, g: dict, consts: dict):
        """Re-validate inlined constants by type and value, adopting the objects now bound to them."""
        import math

        copysign = math.copysign
        specs = []
        for name, (kind, raw) in consts.items():
            typ = {"int": int, "float": float, "bool": bool}[kind]
            value = float.fromhex(raw) if kind == "float" else typ(raw)
            specs.append((name, typ, value, self.cell("const", name)))

        def refresh() -> bool:
            for name, typ, value, cell in specs:
                v = g.get(name, _UNSET)
                if type(v) is not typ or v != value or (typ is float and copysign(1.0, v) != copysign(1.0, value)):
                    return False
                cell.cell_contents = v
            return True

        return refresh


def make_accelerators(manifest: dict) -> dict:
    """Map function name -> decorator replacing it with a guarded native fast path.

    Functions that native code calls but that are not accelerated themselves
    get a decorator too: it returns the function unchanged and records its
    identity, so the callers' fast paths can tell when it has been rebound.
    """
    import ctypes

    _CTYPES = {"int": ctypes.c_int64, "float": ctypes.c_double, "bool": ctypes.c_uint8}
    lib = ctypes.CDLL(manifest["dll"])
    lib.pyr_init.restype = None
    lib.pyr_init()
    take_error = lib.pyr_take_error
    take_error.argtypes = []
    take_error.restype = ctypes.c_int32
    ctx = _Context(take_error)
    builtin_fn = _fast_calls(lib, ctypes)
    accel: dict = {}
    for spec in manifest["functions"]:
        if "deps" not in spec:
            continue  # no dependency information: never run native code unguarded
        variants = spec.get("variants") or [
            {"params": spec["params"], "ret": spec["ret"], "symbol": f"pyr_export_{spec['name']}"}
        ]
        calls = []
        for var in variants:
            fast = builtin_fn(var["symbol"]) if builtin_fn else None
            if fast is not None:
                calls.append((dict(var, protocol="object"), fast))
                continue
            cfunc = getattr(lib, var["symbol"])
            cfunc.argtypes = [_CTYPES[p] for p in var["params"]]
            cfunc.restype = _CTYPES[var["ret"]]
            calls.append((var, cfunc))

        def decorate(orig, _spec=spec, _calls=calls):
            return _wrap(orig, _spec, _calls, ctx)

        accel[spec["name"]] = decorate
    for spec in manifest["functions"]:
        for dep in (spec.get("deps") or {}).get("funcs", ()):
            if dep not in accel:
                accel[dep] = lambda orig, _name=dep: ctx.register(_name, orig)
    return accel


def _fast_calls(lib, ctypes):
    """symbol -> the export as a CPython builtin function (None: use ctypes for it).

    A ctypes call spends most of its time in libffi; a METH_FASTCALL builtin
    costs a fraction of that. The library gets the few C-API functions it
    needs by pointer, so it never links against a particular Python.
    """
    if sys.implementation.name != "cpython":
        return None
    try:
        api = ctypes.pythonapi
        names = ("as_i64", "from_i64", "as_f64", "from_f64", "save_thread", "restore_thread", "incref")
        funcs = ("PyLong_AsLongLong", "PyLong_FromLongLong", "PyFloat_AsDouble", "PyFloat_FromDouble",
                 "PyEval_SaveThread", "PyEval_RestoreThread", "Py_IncRef")

        class PyApi(ctypes.Structure):
            _fields_ = [(n, ctypes.c_void_p) for n in names + ("none", "py_true", "py_false")]

        table = PyApi(*[ctypes.cast(getattr(api, f), ctypes.c_void_p).value for f in funcs], id(None), id(True), id(False))
        lib.pyr_set_api.argtypes = [ctypes.POINTER(PyApi)]
        lib.pyr_set_api.restype = None
        lib.pyr_set_api(ctypes.byref(table))
        new_function = api.PyCFunction_NewEx
        new_function.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
        new_function.restype = ctypes.py_object
    except (AttributeError, OSError, ValueError, TypeError):
        return None

    def make(symbol: str):
        try:
            method_def = getattr(lib, symbol.replace("pyr_export_", "pyr_def_", 1))
        except AttributeError:
            return None
        method_def.argtypes = []
        method_def.restype = ctypes.c_void_p
        return new_function(method_def(), None, None)

    return make


def _relocate(node, lineno: int) -> None:
    import ast

    for sub in ast.walk(node):
        if "lineno" in sub._attributes:
            sub.lineno = sub.end_lineno = lineno
            sub.col_offset = sub.end_col_offset = 0


def _wrap(orig, spec: dict, calls: list, ctx: _Context):
    """`orig` with a guarded native fast path in front of its own body.

    The fast path runs only when the arguments match a compiled signature and
    every global the native code baked in (functions it calls, inlined
    constants, `math` attributes, builtins) is still bound to the object it was
    compiled against. Otherwise, or when the native code bails out, the
    original body runs, so the behaviour is exactly Python's.
    """
    import ast

    name = spec["name"]
    code = orig.__code__
    argnames = list(code.co_varnames[: code.co_argcount])
    if (
        code.co_kwonlyargcount
        or code.co_posonlyargcount
        or code.co_flags & 0x0C
        or any(len(var["params"]) != len(argnames) for var, _ in calls)
    ):
        return ctx.register(name, orig)  # signature changed since planning: leave it alone

    deps = spec.get("deps") or {}
    cells: dict = {}

    def bind(cell_name: str, value) -> None:
        cells[cell_name] = types.CellType(value)

    terms = []
    with_defaults = set(deps.get("defaults", ()))
    for f in deps.get("funcs", ()):
        # still the same function, still running the same code (and defaults) the native call stands for
        terms.append(f"__pyr_g[{f!r}] is __pyr_f_{f} and __pyr_f_{f}.__code__ is __pyr_fc_{f}")
        cells[f"__pyr_f_{f}"] = ctx.cell("fn", f)
        cells[f"__pyr_fc_{f}"] = ctx.cell("code", f)
        if f in with_defaults:
            terms.append(f"__pyr_f_{f}.__defaults__ is __pyr_fd_{f}")
            cells[f"__pyr_fd_{f}"] = ctx.cell("defaults", f)
    consts = deps.get("consts") or {}
    for k in consts:
        terms.append(f"__pyr_g[{k!r}] is __pyr_k_{k}")
        cells[f"__pyr_k_{k}"] = ctx.cell("const", k)
    if deps.get("math") is not None:
        terms.append("__pyr_g['math'] is __pyr_m")
        bind("__pyr_m", ctx.math)
        bind("__pyr_md", getattr(ctx.math, "__dict__", {}))
        for a in deps["math"]:
            terms.append(f"__pyr_md[{a!r}] is __pyr_a_{a}")
            bind(f"__pyr_a_{a}", ctx.math_attr(a))
    for b in deps.get("builtins", ()):
        terms.append(f"{b!r} not in __pyr_g and __pyr_bd[{b!r}] is __pyr_b_{b}")
        bind(f"__pyr_b_{b}", ctx.builtins.get(b, _UNSET))
    if deps.get("builtins"):
        bind("__pyr_bd", orig.__builtins__)
    if terms:
        bind("__pyr_g", orig.__globals__)
    check = " and ".join(terms)
    if consts:
        bind("__pyr_refresh", ctx.refresher(orig.__globals__, consts))
        check = f"({check} or (__pyr_refresh() and {check}))"

    args = ", ".join(argnames)
    lines = []
    for i, (var, cfunc) in enumerate(calls):
        guard = " and ".join(_GUARD[p].format(a=a) for p, a in zip(var["params"], argnames)) or "True"
        lines.append(f"{'if' if i == 0 else 'elif'} {guard}{' and ' + check if check else ''}:")
        lines.append(f"    __pyr_r = __pyr_c{i}({args})")
        lines += ["    " + line for line in _RETURN[var.get("protocol", var["ret"])].splitlines()]
        bind(f"__pyr_c{i}", cfunc)
    if any(var.get("protocol", var["ret"]) in ("int", "float") for var, _ in calls):
        bind("__pyr_err", ctx.take_error)
    guard_src = "\n".join(lines)
    if terms:  # a missing global raises KeyError in the check: run the original
        guard_src = "try:\n" + "\n".join("    " + line for line in lines) + "\nexcept __pyr_KeyError:\n    pass"
    for builtin_name in ("type", "int", "float", "bool", "KeyError"):
        bind(f"__pyr_{builtin_name}", ctx.builtins[builtin_name])

    lineno = code.co_firstlineno
    node = _DEFS.pop((code.co_filename, name), None)
    if node is not None and node.lineno == lineno:
        body = node.body
    else:  # source not recorded: a separate wrapper that calls the original
        bind("__pyr_orig", orig)
        body = ast.parse(f"return __pyr_orig({args})").body
        _relocate(ast.Module(body=body, type_ignores=[]), lineno)
    factory = ast.parse(
        f"def __pyr_factory({', '.join(cells)}):\n    def __pyr_fn({args}):\n        pass\n    return __pyr_fn\n"
    )
    guard_stmts = ast.parse(guard_src).body
    _relocate(factory, lineno)
    _relocate(ast.Module(body=guard_stmts, type_ignores=[]), lineno)
    inner = factory.body[0].body[0]
    doc = []
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        doc, body = body[:1], body[1:]
    inner.body = doc + guard_stmts + body
    module_code = compile(factory, code.co_filename, "exec", dont_inherit=True)
    factory_code = next(c for c in module_code.co_consts if isinstance(c, types.CodeType))
    fn_code = next(c for c in factory_code.co_consts if isinstance(c, types.CodeType) and c.co_name == "__pyr_fn")
    fn_code = fn_code.replace(co_name=orig.__name__, co_qualname=orig.__qualname__)
    w = types.FunctionType(
        fn_code, orig.__globals__, orig.__name__, orig.__defaults__, tuple(cells[n] for n in fn_code.co_freevars)
    )
    w.__qualname__ = orig.__qualname__
    w.__doc__ = orig.__doc__
    w.__module__ = orig.__module__
    w.__kwdefaults__ = orig.__kwdefaults__
    lazy = hasattr(orig, "__annotate__")  # 3.14+: copy the annotate function, do not evaluate it
    for attr in ("__annotate__" if lazy else "__annotations__", "__type_params__"):
        try:
            setattr(w, attr, getattr(orig, attr))
        except (AttributeError, TypeError):
            pass
    w.__dict__.update(orig.__dict__)
    w.__pyrust_native__ = True
    return ctx.register(name, w)


def instrument(source: str, filename: str, names: set[str]) -> types.CodeType:
    """Compile `source`, rebinding each accelerated top-level def right after it is defined.

    Once the last one is rebound, `__pyrust_accel__` is removed from the
    program's globals again.
    """
    if not names:
        return compile(source, filename, "exec", dont_inherit=True)
    import ast
    import copy

    tree = ast.parse(source, filename=filename)
    body: list[ast.stmt] = []
    last = None
    for stmt in tree.body:
        body.append(stmt)
        if isinstance(stmt, ast.FunctionDef) and stmt.name in names:
            _DEFS[(filename, stmt.name)] = copy.deepcopy(stmt)
            rebind = ast.Assign(
                targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Subscript(
                        value=ast.Name(id="__pyrust_accel__", ctx=ast.Load()),
                        slice=ast.Constant(value=stmt.name),
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=stmt.name, ctx=ast.Load())],
                    keywords=[],
                ),
            )
            body.append(ast.copy_location(rebind, stmt))
            last = len(body)
    if last is not None:
        cleanup = ast.Try(
            body=[ast.Delete(targets=[ast.Name(id="__pyrust_accel__", ctx=ast.Del())])],
            handlers=[
                ast.ExceptHandler(type=ast.Name(id="NameError", ctx=ast.Load()), name=None, body=[ast.Pass()])
            ],
            orelse=[],
            finalbody=[],
        )
        body.insert(last, ast.copy_location(cleanup, body[last - 1]))
    tree.body = body
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec", dont_inherit=True)


class _PyrLoader(importlib.machinery.SourceFileLoader):
    """Imports `.pyr` files as ordinary Python modules (no bytecode cache)."""

    def set_data(self, path, data, **kwargs):  # noqa: D401 - never write .pyc for .pyr
        return None


def install_pyr_import_hook() -> None:
    m = importlib.machinery
    hook = m.FileFinder.path_hook(
        (m.ExtensionFileLoader, m.EXTENSION_SUFFIXES),
        (_PyrLoader, [".pyr"]),
        (m.SourceFileLoader, m.SOURCE_SUFFIXES),
        (m.SourcelessFileLoader, m.BYTECODE_SUFFIXES),
    )
    for i, existing in enumerate(sys.path_hooks):
        if "FileFinder" in repr(existing):
            sys.path_hooks[i] = hook
            break
    else:
        sys.path_hooks.append(hook)
    sys.path_importer_cache.clear()


def _decode_source(data: bytes) -> str | bytes:
    """Source text decoded exactly as Python decodes a script (BOM, PEP 263 cookie).

    Undecodable input is returned as bytes so that `compile` raises the same
    SyntaxError Python itself would.
    """
    head_end = data.find(b"\n", data.find(b"\n") + 1)
    head = data if head_end < 0 else data[:head_end]
    if data.startswith(b"\xef\xbb\xbf") or b"coding" in head:
        from importlib.util import decode_source  # slower (tokenize), so only when needed

        return decode_source(data)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data


def _strip_runner_frames(tb):
    """Hide this runner's own frames so tracebacks look like plain Python's."""
    while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
        tb = tb.tb_next
    return tb


def _report_uncaught(exc: BaseException, tb) -> None:
    """Print an uncaught exception the way the interpreter does: through sys.excepthook."""
    exc.__traceback__ = tb
    sys.last_exc = exc
    sys.last_type, sys.last_value, sys.last_traceback = type(exc), exc, tb
    hook = getattr(sys, "excepthook", None)
    if hook is None:
        print("sys.excepthook is missing", file=sys.stderr)
        sys.__excepthook__(type(exc), exc, tb)
        return
    try:
        hook(type(exc), exc, tb)
    except BaseException as hook_exc:  # noqa: BLE001 - mirror the interpreter's fallback
        print("Error in sys.excepthook:", file=sys.stderr)
        sys.__excepthook__(type(hook_exc), hook_exc, _strip_runner_frames(hook_exc.__traceback__))
        print("\nOriginal exception was:", file=sys.stderr)
        sys.__excepthook__(type(exc), exc, tb)


_sigint_exit: tuple | None = None  # (os, signal), bound before the program's imports


def _exit_like_sigint() -> None:
    """atexit hook: after an uncaught KeyboardInterrupt, die by SIGINT as Python does."""
    if _sigint_exit is None:
        return
    os, signal = _sigint_exit
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)


def _make_main_module(path: str, directory_entry: bool) -> types.ModuleType:
    """A `__main__` module with the attributes `python script.py` / `python dir` gives it."""
    import builtins

    mod = types.ModuleType("__main__")
    loader = importlib.machinery.SourceFileLoader("__main__", path)
    mod.__loader__ = loader
    if directory_entry:
        mod.__package__ = ""
        mod.__spec__ = importlib.machinery.ModuleSpec("__main__", loader, origin=path)
    mod.__dict__["__builtins__"] = builtins
    mod.__file__ = path
    cached = None
    if directory_entry and path.endswith(".py"):
        try:
            from importlib.util import cache_from_source

            cached = cache_from_source(path)
        except (NotImplementedError, ValueError):
            cached = None
    mod.__cached__ = cached
    return mod


def main(
    argv: list[str],
    *,
    argv0: str | None = None,
    main_file: str | None = None,
    directory_entry: bool = False,
    quiet: bool = False,
    isolate: bool = False,
) -> int:
    """Run `argv[0]` (the script) as `__main__`. argv: [script, manifest|-, script_dir, *args].

    argv0 / main_file: the program's sys.argv[0] and __file__ (default: the script path).
    isolate: this process exists only to run the program, so drop every module
    PyRust imported (the program must resolve imports exactly as under `python`)
    and exit like the interpreter on an uncaught KeyboardInterrupt.
    """
    script, manifest_path, script_dir, args = argv[0], argv[1], argv[2], argv[3:]
    import os

    main_file = main_file or script
    with open(script, "rb") as fh:
        data = fh.read()

    accel: dict = {}
    if manifest_path != "-":
        try:
            import json

            with open(manifest_path, encoding="utf-8") as fh:
                accel = make_accelerators(json.load(fh))
        except Exception as exc:  # a broken native library must never break the program
            if not quiet:
                reason = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
                if len(reason) > 160:
                    reason = reason[:157] + "..."
                print(f"[pyrust] native functions unavailable ({reason}); running on CPython", file=sys.stderr)
            accel = {}

    # Compile before the module reset below: decoding and instrumenting may
    # import stdlib modules, which must not resolve to the program's own files.
    code = syntax_error = None
    try:
        code = instrument(_decode_source(data), main_file, set(accel))
    except SyntaxError as exc:
        syntax_error = exc

    sys.argv = [argv0 if argv0 is not None else script, *args]
    if sys.path:
        sys.path[0] = script_dir
    else:
        sys.path.insert(0, script_dir)
    install_pyr_import_hook()

    mod = _make_main_module(main_file, directory_entry)
    if accel:
        # Looked up by the instrumented module code; kept out of the program's globals().
        import builtins

        builtins.__pyrust_accel__ = accel
    if isolate:
        import atexit
        import signal

        global _sigint_exit
        sigint_tools = (os, signal)
        atexit.register(_exit_like_sigint)  # registered first, so it runs after the program's own hooks
        from pyrust import restore_modules

        restore_modules()
    sys.modules["__main__"] = mod
    if syntax_error is not None:
        _report_uncaught(syntax_error, None)  # like Python: no traceback for a script that does not compile
        return 1
    try:
        exec(code, mod.__dict__)
    except SystemExit:
        raise
    except BaseException as exc:
        _report_uncaught(exc, _strip_runner_frames(exc.__traceback__))
        if isinstance(exc, KeyboardInterrupt):
            if os.name == "nt":
                return 0xC000013A - (1 << 32)  # STATUS_CONTROL_C_EXIT, like python.exe
            if isolate:
                _sigint_exit = sigint_tools
            return 130
        return 1
    return 0
