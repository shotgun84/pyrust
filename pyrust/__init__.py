"""PyRust: run Python at native speed where possible, as plain Python everywhere else."""

import sys as _sys

# The modules the interpreter had loaded before PyRust imported anything of its
# own. A program run on CPython by `pyrust run` gets exactly this set back, so
# its imports resolve as under plain `python` (a local json.py must win over
# the stdlib json that the transpiler happened to import first). `runpy` is
# only there because of `python -m pyrust.cli`.
_BASELINE_MODULES = frozenset(_sys.modules) - {"pyrust", "runpy"}

__version__ = "0.2.0"
__all__ = [
    "__version__",
    "transpile_source",
    "emit_cargo_project",
    "build_project",
    "run_project",
    "plan_execution",
]

_LAZY = {
    "transpile_source": "pyrust.codegen",
    "emit_cargo_project": "pyrust.builder",
    "build_project": "pyrust.builder",
    "run_project": "pyrust.builder",
    "plan_execution": "pyrust.builder",
}


def restore_modules() -> None:
    """Forget every module imported since the interpreter started PyRust."""
    modules = _sys.modules
    for name in [n for n in modules if n not in _BASELINE_MODULES]:
        if name != "__main__":
            del modules[name]


def __getattr__(name: str):
    # Importing the transpiler costs ~100 ms; a cached `pyrust run` never needs it.
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'pyrust' has no attribute {name!r}")
