import pytest

from pyrust.errors import PyRustError
from pyrust.types import (
    ANY,
    BOOL,
    FLOAT,
    INT,
    NONE,
    STR,
    UNKNOWN,
    ClassType,
    DictType,
    FuncType,
    ListType,
    SetType,
    TupleType,
    type_from_annotation_name,
    unify_types,
)


def test_primitive_rust_type_names():
    assert INT.to_rust_type() == "i64"
    assert FLOAT.to_rust_type() == "f64"
    assert BOOL.to_rust_type() == "bool"
    assert STR.to_rust_type() == "String"
    assert NONE.to_rust_type() == "()"
    assert ANY.to_rust_type() == "PyValue"
    assert UNKNOWN.to_rust_type() == "PyValue"


def test_primitive_copy_semantics():
    assert INT.is_copy is True
    assert FLOAT.is_copy is True
    assert BOOL.is_copy is True
    assert NONE.is_copy is True
    assert STR.is_copy is False
    assert ANY.is_copy is False
    assert ListType(INT).is_copy is False


def test_unify_identical_and_unknown_types():
    assert unify_types(UNKNOWN, INT) == INT
    assert unify_types(STR, UNKNOWN) == STR
    assert unify_types(INT, INT) == INT
    assert unify_types(STR, STR) == STR


def test_unify_numeric_kinds_stay_dynamic():
    # A value that is an int on one path and a float on another keeps its own
    # type in Python (`0` prints as `0`), so merging must not promote.
    assert unify_types(INT, FLOAT) == ANY
    assert unify_types(FLOAT, INT) == ANY
    assert unify_types(BOOL, INT) == ANY


def test_arith_result_type_follows_python():
    from pyrust.types import arith_result_type

    assert arith_result_type("Add", INT, FLOAT) == FLOAT
    assert arith_result_type("Add", BOOL, BOOL) == INT
    assert arith_result_type("BitAnd", BOOL, BOOL) == BOOL
    assert arith_result_type("Div", INT, INT) == FLOAT
    assert arith_result_type("BitOr", INT, FLOAT) is None


def test_unify_incompatible_types_promotes_to_pyvalue():
    assert unify_types(INT, STR) == ANY
    assert unify_types(FLOAT, STR) == ANY
    assert unify_types(ListType(INT), DictType(STR, INT)) == ANY


def test_unify_container_types():
    assert unify_types(ListType(UNKNOWN), ListType(INT)) == ListType(INT)
    assert unify_types(ListType(INT), ListType(FLOAT)) == ListType(ANY)
    assert unify_types(ListType(INT), ListType(STR)) == ListType(ANY)
    assert unify_types(ListType(INT), None) == ListType(INT)

    d1 = DictType(UNKNOWN, UNKNOWN)
    d2 = DictType(STR, INT)
    assert unify_types(d1, d2) == DictType(STR, INT)
    assert unify_types(DictType(STR, INT), DictType(STR, STR)) == DictType(STR, ANY)

    s1 = SetType(UNKNOWN)
    s2 = SetType(STR)
    assert unify_types(s1, s2) == SetType(STR)
    assert SetType(STR).to_rust_type() == "std::collections::HashSet<String>"


def test_tuple_and_class_rust_representation():
    tup = TupleType((INT, STR))
    assert tup.to_rust_type() == "(i64, String)"
    assert unify_types(TupleType((INT, STR)), TupleType((FLOAT, STR))) == TupleType((ANY, STR))
    assert unify_types(TupleType((INT,)), TupleType((INT, INT))) == ANY

    cls = ClassType(
        name="Point",
        fields={"x": FLOAT, "y": FLOAT},
        methods={"norm": FuncType(params=[], ret=FLOAT)},
    )
    assert cls.to_rust_type() == "Point"
    assert cls.is_copy is False


def test_type_from_annotation_name():
    assert type_from_annotation_name("int") == INT
    assert type_from_annotation_name("float") == FLOAT
    assert type_from_annotation_name("str") == STR
    assert type_from_annotation_name("bool") == BOOL
    assert type_from_annotation_name("None") == NONE
    assert type_from_annotation_name("Any") == ANY


def test_pyrust_error_formatting():
    err = PyRustError("Unsupported construct", lineno=12, col=4, filename="script.pyr", source_line="    yield x")
    text = str(err)
    assert "script.pyr:12:4" in text
    assert "Unsupported construct" in text
    assert "yield x" in text
