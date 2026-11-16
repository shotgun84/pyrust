"""PyRust: run Python at native speed where possible, as plain Python everywhere else."""

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


def __getattr__(name: str):
    # Importing the transpiler costs ~100 ms; a cached `pyrust run` never needs it.
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'pyrust' has no attribute {name!r}")
