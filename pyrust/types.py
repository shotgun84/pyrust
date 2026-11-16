from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PyType:
    """Base type in the PyRust inference lattice."""

    @property
    def is_copy(self) -> bool:
        return False

    def to_rust_type(self) -> str:
        return "PyValue"


@dataclass(frozen=True)
class PrimitiveType(PyType):
    name: str
    rust_name: str
    copy: bool = False

    @property
    def is_copy(self) -> bool:
        return self.copy

    def to_rust_type(self) -> str:
        return self.rust_name


UNKNOWN = PrimitiveType("unknown", "PyValue", copy=False)
NONE = PrimitiveType("None", "()", copy=True)
BOOL = PrimitiveType("bool", "bool", copy=True)
INT = PrimitiveType("int", "i64", copy=True)
FLOAT = PrimitiveType("float", "f64", copy=True)
STR = PrimitiveType("str", "String", copy=False)
ANY = PrimitiveType("Any", "PyValue", copy=False)


@dataclass(frozen=True)
class ListType(PyType):
    elem: PyType = UNKNOWN

    def to_rust_type(self) -> str:
        elem_rust = self.elem.to_rust_type() if self.elem != UNKNOWN else "PyValue"
        return f"Vec<{elem_rust}>"


@dataclass(frozen=True)
class DictType(PyType):
    key: PyType = UNKNOWN
    val: PyType = UNKNOWN

    def to_rust_type(self) -> str:
        k_rust = self.key.to_rust_type() if self.key != UNKNOWN else "String"
        v_rust = self.val.to_rust_type() if self.val != UNKNOWN else "PyValue"
        return f"PyDict<{k_rust}, {v_rust}>"


@dataclass(frozen=True)
class SetType(PyType):
    elem: PyType = UNKNOWN

    def to_rust_type(self) -> str:
        elem_rust = self.elem.to_rust_type() if self.elem != UNKNOWN else "PyValue"
        return f"std::collections::HashSet<{elem_rust}>"


@dataclass(frozen=True)
class TupleType(PyType):
    elements: tuple[PyType, ...] = ()

    @property
    def is_copy(self) -> bool:
        return all(e.is_copy for e in self.elements)

    def to_rust_type(self) -> str:
        if not self.elements:
            return "()"
        inner = ", ".join(e.to_rust_type() for e in self.elements)
        return f"({inner})"


@dataclass(frozen=True)
class FuncType(PyType):
    params: list[PyType] = field(default_factory=list)
    ret: PyType = UNKNOWN
    has_vararg: bool = False

    def to_rust_type(self) -> str:
        params_str = ", ".join(p.to_rust_type() for p in self.params)
        return f"fn({params_str}) -> {self.ret.to_rust_type()}"


@dataclass(frozen=True)
class ClassType(PyType):
    name: str
    fields: dict[str, PyType] = field(default_factory=dict)
    methods: dict[str, FuncType] = field(default_factory=dict)

    def to_rust_type(self) -> str:
        return self.name


def _merge_class_types(a: ClassType, b: ClassType) -> ClassType:
    """Merge two inferred views of the same-named class (fields may differ per branch)."""
    merged_fields: dict[str, PyType] = {}
    for key in list(a.fields.keys()) + [k for k in b.fields.keys() if k not in a.fields]:
        merged_fields[key] = unify_container_elem(
            a.fields.get(key, UNKNOWN), b.fields.get(key, UNKNOWN)
        )
    merged_methods = dict(a.methods)
    for key, value in b.methods.items():
        merged_methods.setdefault(key, value)
    return ClassType(a.name, merged_fields, merged_methods)


def unify_container_elem(a: PyType, b: PyType) -> PyType:
    """Unify container element types without silent coercion.

    Python containers are heterogeneous: `[1, 2.5]` keeps an int `1` (prints
    as `1`, serializes as `1`). Any two element types that are not identical
    therefore promote to dynamic PyValue instead of being coerced.
    """
    if a == UNKNOWN:
        return b
    if b == UNKNOWN:
        return a
    if isinstance(a, ClassType) and isinstance(b, ClassType) and a.name == b.name:
        return _merge_class_types(a, b)
    if isinstance(a, ListType) and isinstance(b, ListType):
        return ListType(unify_container_elem(a.elem, b.elem))
    if isinstance(a, SetType) and isinstance(b, SetType):
        return SetType(unify_container_elem(a.elem, b.elem))
    if isinstance(a, DictType) and isinstance(b, DictType):
        return DictType(
            unify_container_elem(a.key, b.key),
            unify_container_elem(a.val, b.val),
        )
    if isinstance(a, TupleType) and isinstance(b, TupleType) and len(a.elements) == len(b.elements):
        return TupleType(tuple(unify_container_elem(x, y) for x, y in zip(a.elements, b.elements)))
    if a == b:
        return a
    return ANY


def unify_types(t1: PyType | None, t2: PyType | None) -> PyType:
    """Least upper bound of two types in the PyRust lattice.

    Incompatible concrete types unify to ANY (which compiles to the Rust `PyValue` enum).
    """
    if t1 is None or t1 == UNKNOWN:
        return t2 if t2 is not None else UNKNOWN
    if t2 is None or t2 == UNKNOWN:
        return t1
    if t1 == t2:
        return t1
    if t1 == ANY or t2 == ANY:
        return ANY

    # No numeric promotion here: a variable/parameter/result that is an int on
    # one path and a float (or bool) on another keeps its runtime type in
    # Python (`0` prints as `0`, not `0.0`), so it must stay dynamic. Mixed
    # *arithmetic* is typed separately (see `arith_result_type`).

    # Containers unify recursively (bool is preserved by unify_container_elem).
    if isinstance(t1, ListType) and isinstance(t2, ListType):
        return ListType(unify_container_elem(t1.elem, t2.elem))

    if isinstance(t1, SetType) and isinstance(t2, SetType):
        return SetType(unify_container_elem(t1.elem, t2.elem))

    if isinstance(t1, DictType) and isinstance(t2, DictType):
        return DictType(
            unify_container_elem(t1.key, t2.key),
            unify_container_elem(t1.val, t2.val),
        )

    if isinstance(t1, TupleType) and isinstance(t2, TupleType):
        if len(t1.elements) != len(t2.elements):
            return ANY
        return TupleType(tuple(unify_container_elem(a, b) for a, b in zip(t1.elements, t2.elements)))

    if isinstance(t1, ClassType) and isinstance(t2, ClassType):
        if t1.name == t2.name:
            return _merge_class_types(t1, t2)
        return ANY

    # Any other mismatch falls back to dynamic PyValue
    return ANY


def type_from_annotation_name(name: str) -> PyType:
    mapping: dict[str, PyType] = {
        "int": INT,
        "i64": INT,
        "float": FLOAT,
        "f64": FLOAT,
        "bool": BOOL,
        "str": STR,
        "String": STR,
        "None": NONE,
        "Any": ANY,
        "object": ANY,
        "list": ListType(UNKNOWN),
        "dict": DictType(UNKNOWN, UNKNOWN),
        "set": SetType(UNKNOWN),
    }
    return mapping.get(name, UNKNOWN)


def arith_result_type(op_name: str, left: PyType, right: PyType) -> PyType | None:
    """Result type of a numeric binary operator, or None if not numeric.

    Mirrors Python: bool behaves as int in arithmetic (True + True == 2),
    except for &, |, ^ between two bools, which stay bool; anything mixed
    with a float is a float; `/` always yields a float.
    """
    nums = (BOOL, INT, FLOAT)
    if left not in nums or right not in nums:
        return None
    if op_name == "Div":
        return FLOAT
    if op_name in ("BitAnd", "BitOr", "BitXor"):
        if left == BOOL and right == BOOL:
            return BOOL
        if FLOAT in (left, right):
            return None  # TypeError in Python
        return INT
    if op_name in ("LShift", "RShift"):
        return None if FLOAT in (left, right) else INT
    if FLOAT in (left, right):
        return FLOAT
    return INT
