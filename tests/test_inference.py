from pathlib import Path
import pytest

from pyrust.errors import PyRustError
from pyrust.inference import ModuleTypeEnv, ScopeInfo, infer_module
from pyrust.parser import ParsedModule, load_project, parse_source
from pyrust.types import ANY, BOOL, FLOAT, INT, NONE, STR, ClassType, DictType, FuncType, ListType


def test_parse_source_separates_defs_classes_and_main_guard():
    source = """
import math
from utils import helper

class Counter:
    def __init__(self, start):
        self.val = start

def compute(x):
    return x * 2

top_val = compute(21)

if __name__ == "__main__":
    print(top_val)
"""
    parsed = parse_source(source, filename="demo.pyr", module_name="demo")
    assert isinstance(parsed, ParsedModule)
    assert parsed.name == "demo"
    assert [f.name for f in parsed.functions] == ["compute"]
    assert [c.name for c in parsed.classes] == ["Counter"]
    assert "math" in parsed.imports
    assert "utils" in parsed.imports
    # Both top_val = compute(21) and print(top_val) inside __main__ guard are lifted into main_body
    assert len(parsed.main_body) == 2


def test_parse_source_syntax_error_raises_pyrust_error():
    with pytest.raises(PyRustError) as exc_info:
        parse_source("def broken(:\n    pass\n", filename="broken.pyr")
    assert "broken.pyr" in str(exc_info.value)


def test_load_project_discovers_pyr_and_py_files(tmp_path: Path):
    (tmp_path / "main.pyr").write_text("import helpers\nprint(helpers.double(5))\n", encoding="utf-8")
    (tmp_path / "helpers.py").write_text("def double(x):\n    return x * 2\n", encoding="utf-8")

    modules = load_project(tmp_path)
    names = [m.name for m in modules]
    assert names[0] == "main"
    assert "helpers" in names


def test_infer_untyped_literals_and_arithmetic():
    src = """
a = 10
b = 2.5
c = a + b
d = a // 3
s = "hello"
flag = a > 5
items = [1, 2, 3]
mapping = {"k": 42}
"""
    env = infer_module(parse_source(src))
    v = env.module_scope.var_types
    assert v["a"] == INT
    assert v["b"] == FLOAT
    assert v["c"] == FLOAT
    assert v["d"] == INT
    assert v["s"] == STR
    assert v["flag"] == BOOL
    assert v["items"] == ListType(INT)
    assert v["mapping"] == DictType(STR, INT)


def test_infer_untyped_functions_from_callsites_and_recursion():
    src = """
def add(x, y):
    return x + y

def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

total = add(10, 20)
f10 = fib(10)
"""
    env = infer_module(parse_source(src))
    assert env.func_types["add"] == FuncType(params=[INT, INT], ret=INT)
    assert env.func_types["fib"] == FuncType(params=[INT], ret=INT)
    assert env.module_scope.var_types["total"] == INT
    assert env.module_scope.var_types["f10"] == INT


def test_infer_dynamic_reassignment_promotes_to_pyvalue_and_marks_mutated():
    src = """
val = 100
val = "switched to string"
"""
    env = infer_module(parse_source(src))
    assert env.module_scope.var_types["val"] == ANY
    assert "val" in env.module_scope.mutated_vars


def test_infer_hoisted_variables_assigned_inside_blocks():
    src = """
def choose(flag):
    if flag:
        answer = 42
    else:
        answer = 99
    return answer

out = choose(True)
"""
    env = infer_module(parse_source(src))
    fn_scope = env.func_scopes["choose"]
    assert fn_scope.var_types["answer"] == INT
    assert "answer" in fn_scope.hoisted_vars
    assert "answer" in fn_scope.mutated_vars


def test_infer_class_fields_and_methods_without_annotations():
    src = """
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def length_sq(self):
        return self.x * self.x + self.y * self.y

v = Vector2D(3.0, 4.0)
lsq = v.length_sq()
"""
    env = infer_module(parse_source(src))
    cls = env.class_types["Vector2D"]
    assert isinstance(cls, ClassType)
    assert cls.fields["x"] == FLOAT
    assert cls.fields["y"] == FLOAT
    assert cls.methods["length_sq"].ret == FLOAT
    assert env.module_scope.var_types["v"] == cls
    assert env.module_scope.var_types["lsq"] == FLOAT

