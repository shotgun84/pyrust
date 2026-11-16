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


def main(argv: list[str]) -> int:
    script, manifest_path, script_dir, args = argv[0], argv[1], argv[2], argv[3:]
    with open(script, encoding="utf-8") as fh:
        source = fh.read()

    accel: dict = {}
    if manifest_path != "-":
        try:
            import json

            with open(manifest_path, encoding="utf-8") as fh:
                accel = make_accelerators(json.load(fh))
        except Exception as exc:  # a broken native library must never break the program
            print(f"[pyrust] native functions unavailable ({exc}); running on CPython", file=sys.stderr)
            accel = {}

    sys.argv = [script, *args]
    sys.path[0] = script_dir
    install_pyr_import_hook()

    mod = types.ModuleType("__main__")
    mod.__file__ = script
    mod.__dict__["__builtins__"] = __builtins__
    if accel:
        mod.__dict__["__pyrust_accel__"] = accel
    sys.modules["__main__"] = mod
    try:
        code = instrument(source, script, set(accel))
        exec(code, mod.__dict__)
    except SystemExit:
        raise
    except BaseException as exc:
        tb = exc.__traceback__
        # Hide this runner's own frame so tracebacks look like plain Python's.
        while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
            tb = tb.tb_next
        import traceback

        traceback.print_exception(type(exc), exc, tb)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    return 0
