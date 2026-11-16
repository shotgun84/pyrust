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
    "int": "type({a}) is int and -9223372036854775808 <= {a} <= 9223372036854775807",
    "float": "type({a}) is float",
    "bool": "type({a}) is bool",
}


def make_accelerators(manifest: dict) -> dict:
    """Map function name -> decorator replacing it with a guarded native fast path."""
    import ctypes

    _CTYPES = {"int": ctypes.c_int64, "float": ctypes.c_double, "bool": ctypes.c_uint8}
    lib = ctypes.CDLL(manifest["dll"])
    lib.pyr_init.restype = None
    lib.pyr_init()
    take_error = lib.pyr_take_error
    take_error.argtypes = []
    take_error.restype = ctypes.c_int32
    accel: dict = {}
    for spec in manifest["functions"]:
        cfunc = getattr(lib, f"pyr_export_{spec['name']}")
        cfunc.argtypes = [_CTYPES[p] for p in spec["params"]]
        cfunc.restype = _CTYPES[spec["ret"]]

        def decorate(orig, _spec=spec, _c=cfunc):
            return _wrap(orig, _spec, _c, take_error)

        accel[spec["name"]] = decorate
    return accel


# How each return type tells "native result" from "native code bailed out".
# The native side returns a sentinel on failure and raises a per-thread flag,
# so the common case costs one comparison and no extra foreign call.
_RETURN = {
    "int": "        if __pyr_r != -9223372036854775808 or not __pyr_err():\n            return __pyr_r\n",
    "float": "        if __pyr_r == __pyr_r or not __pyr_err():\n            return __pyr_r\n",
    "bool": "        if __pyr_r < 2:\n            return __pyr_r == 1\n",
}


def _wrap(orig, spec: dict, cfunc, take_error):
    """Build the fast-path wrapper with the original's exact signature."""
    code = orig.__code__
    names = list(code.co_varnames[: code.co_argcount])
    defaults = orig.__defaults__ or ()
    first_default = len(names) - len(defaults)
    if len(names) != len(spec["params"]) or code.co_kwonlyargcount or code.co_flags & 0x0C:
        return orig  # signature changed since planning: leave the function alone
    sig = ", ".join(
        n if i < first_default else f"{n}=__pyr_d[{i - first_default}]" for i, n in enumerate(names)
    )
    guards = " and ".join(_GUARD[p].format(a=n) for p, n in zip(spec["params"], names))
    src = (
        "def __pyr_make(__pyr_orig, __pyr_c, __pyr_err, __pyr_d):\n"
        f"    def {spec['name']}({sig}):\n"
        f"        if {guards or 'True'}:\n"
        f"            __pyr_r = __pyr_c({', '.join(names)})\n"
        + _RETURN[spec["ret"]].replace("        ", "            ")
        + f"        return __pyr_orig({', '.join(names)})\n"
        f"    return {spec['name']}\n"
    )
    ns: dict = {}
    exec(src, ns)
    w = ns["__pyr_make"](orig, cfunc, take_error, defaults)
    for attr in ("__name__", "__qualname__", "__doc__", "__module__", "__annotations__"):
        try:
            setattr(w, attr, getattr(orig, attr))
        except (AttributeError, TypeError):
            pass
    w.__dict__.update(getattr(orig, "__dict__", {}))
    w.__wrapped__ = orig
    w.__pyrust_native__ = True
    return w


def instrument(source: str, filename: str, names: set[str]) -> types.CodeType:
    """Compile `source`, rebinding each accelerated top-level def right after it is defined."""
    if not names:
        return compile(source, filename, "exec", dont_inherit=True)
    import ast

    tree = ast.parse(source, filename=filename)
    body: list[ast.stmt] = []
    for stmt in tree.body:
        body.append(stmt)
        if isinstance(stmt, ast.FunctionDef) and stmt.name in names:
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
