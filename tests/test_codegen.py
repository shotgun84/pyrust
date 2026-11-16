import pytest

from pyrust.codegen import generate_rust_module, transpile_source
from pyrust.errors import PyRustError
from pyrust.runtime_template import RUNTIME_RS


def test_runtime_template_contains_pyvalue_and_python_semantics_helpers():
    assert "pub enum PyValue" in RUNTIME_RS
    assert "pub fn py_floordiv" in RUNTIME_RS
    assert "pub fn py_mod" in RUNTIME_RS
    assert "pub fn py_pow" in RUNTIME_RS
    assert "pub fn py_index" in RUNTIME_RS
    # py_random() must scale 53 mantissa bits by 2^-53 for a true [0, 1) range
    assert "9007199254740992.0" in RUNTIME_RS
    assert "pub fn py_randint" in RUNTIME_RS
    assert "pub fn py_shuffle" in RUNTIME_RS
    assert "pub fn py_slice" in RUNTIME_RS


def test_codegen_range_loops_and_step():
    src = """
total = 0
for i in range(0, 10, 2):
    total += i
print(total)
"""
    rust = transpile_source(src)
    assert "let mut total: i64 = 0;" in rust
    assert "for i in (0..10).step_by(2)" in rust
    assert "total += i;" in rust
    assert "println!" in rust


def test_codegen_negative_indexing_and_python_math_ops():
    src = """
nums = [10, 20, 30]
last = nums[-1]
q = 7 // 2
r = -7 % 3
p = 2 ** 8
"""
    rust = transpile_source(src)
    assert "py_index(&nums, (-1))" in rust
    assert "py_floordiv(7, 2)" in rust
    assert "py_mod((-7), 3)" in rust
    assert "py_pow(2, 8)" in rust


def test_codegen_fstrings_and_list_comprehensions():
    src = """
nums = [1, -2, 3, 4]
doubled = [x * 2 for x in nums if x > 0]
msg = f"count={len(doubled)}"
print(msg)
"""
    rust = transpile_source(src)
    assert ".iter()" in rust
    assert ".filter(" in rust
    assert ".map(" in rust
    assert ".collect::<Vec<_>>()" in rust
    assert 'format!("count={}", (doubled.len() as i64))' in rust


def test_codegen_class_struct_and_impl():
    src = """
class Particle:
    def __init__(self, x, v):
        self.x = x
        self.v = v

    def step(self, dt):
        self.x = self.x + self.v * dt
        return self.x

p = Particle(1.0, 2.5)
pos = p.step(0.5)
print(pos)
"""
    rust = transpile_source(src)
    assert "#[derive(Debug, Clone, Default, PartialEq)]" in rust
    assert "pub struct Particle" in rust
    assert "pub x: f64," in rust
    assert "pub v: f64," in rust
    assert "pub fn new(x: f64, v: f64) -> Self" in rust
    assert "pub fn step(&mut self, dt: f64) -> f64" in rust
    assert "let mut p: Particle = Particle::new(1.0f64, 2.5f64);" in rust


def test_codegen_dynamic_type_reassignment_uses_pyvalue():
    src = """
x = 10
x = "hello"
print(x)
"""
    rust = transpile_source(src)
    assert "let mut x: PyValue = PyValue::from(10);" in rust
    assert 'x = PyValue::from("hello".to_string());' in rust


def test_codegen_hoists_block_scoped_variables():
    src = """
def pick(flag):
    if flag:
        val = 10
    else:
        val = 20
    return val

print(pick(True))
"""
    rust = transpile_source(src)
    # `let mut val: i64 = Default::default();` is hoisted before `if flag`
    assert "let mut val: i64 = Default::default();" in rust
    assert "val = 10;" in rust
    assert "val = 20;" in rust


def test_codegen_clones_non_copy_variables_used_multiple_times():
    src = """
def greet(name):
    return "Hi " + name

user = "Alice"
a = greet(user)
b = greet(user)
"""
    rust = transpile_source(src)
    assert "greet(user.clone())" in rust
    assert "greet(user)" in rust


def test_review_fix_1_none_assigned_to_pyvalue():
    src = """
x = None
x = 42
"""
    rust = transpile_source(src)
    assert "PyValue::None" in rust
    assert "PyValue::from(())" not in rust


def test_review_fix_2_negative_index_assign_and_augassign_subscript():
    src = """
items = [1, 2, 3]
items[-1] = 99
items[0] += 5
d = {"k": 10}
d["k"] += 1
"""
    rust = transpile_source(src)
    assert "py_set_index(&mut items, (-1), 99);" in rust
    assert "py_set_index(&mut items, 0, (py_index(&items, 0) + 5));" in rust
    assert 'd.insert("k".to_string(), (d[&"k".to_string()].clone() + 1));' in rust


def test_review_fix_3_string_and_tuple_negative_indexing_and_slicing():
    src = """
s = "hello"
ch = s[-1]
sub = s[1:4]
tup = (10, 20, 30)
last_t = tup[-1]
"""
    rust = transpile_source(src)
    assert "py_str_index(&s, (-1))" in rust
    assert "py_str_slice(&s, Some(1), Some(4), 1)" in rust
    assert "tup.2" in rust


def test_review_fix_4_and_5_ownership_borrow_after_call_and_mut_pop_and_self_field_clone():
    src = """
class Named:
    def __init__(self, label):
        self.label = label

    def get_label(self):
        return self.label

def take_list(xs):
    return len(xs)

items = [1, 2, 3]
n = take_list(items)
first = items[0]
popped = items.pop()
"""
    rust = transpile_source(src)
    assert "let mut items: Vec<i64>" in rust
    assert "take_list(items.clone())" in rust
    assert "return self.label.clone();" in rust


def test_review_fix_6_init_branching_and_partial_tuple_unpack():
    src = """
class Config:
    def __init__(self, flag):
        if flag:
            self.mode = "fast"
        else:
            self.mode = "safe"

a = 1
a, b = 2, 3
"""
    rust = transpile_source(src)
    assert "#[derive(Debug, Clone, Default, PartialEq)]" in rust
    assert 'self_.mode = "fast".to_string();' in rust
    assert "let mut b: i64 = Default::default();" in rust


def test_review_fix_7_module_constants_in_functions_and_cross_module_returns(tmp_path):
    from pyrust.builder import emit_cargo_project

    src = """
SCALE = 10

def scaled(x):
    return x * SCALE

print(scaled(5))
"""
    rust = transpile_source(src)
    assert "const SCALE: i64 = 10;" in rust

    pkg = tmp_path / "cross_pkg"
    pkg.mkdir()
    (pkg / "main.pyr").write_text("import helper\nres = helper.triple(4)\nprint(res)\n", encoding="utf-8")
    (pkg / "helper.py").write_text("def triple(n):\n    return n * 3\n", encoding="utf-8")
    out = tmp_path / "cross_out"
    emit_cargo_project(pkg, out)
    main_rs = (out / "src" / "main.rs").read_text(encoding="utf-8")
    assert "let res: i64 = crate::helper::triple(4);" in main_rs


def test_review_fix_8_pyvalue_comparisons_and_floordiv_pow():
    src = """
x = 10
x = "dyn"
if x == 10:
    print(x)
y = x // 2
z = x ** 3
"""
    rust = transpile_source(src)
    assert "(x.clone() == PyValue::from(10))" in rust or "(x == PyValue::from(10))" in rust
    assert "py_val_floordiv(" in rust
    assert "py_val_pow(" in rust


def test_dep_math_calls_on_float_literals_emit_typed_consts():
    src = """
import math

r = math.sqrt(16.0)
f = math.floor(3.7)
"""
    rust = transpile_source(src)
    assert 'py_math1("sqrt", 16.0f64)' in rust
    assert "py_float_to_int(f64::floor(3.7f64))" in rust


def test_dep_math_constants_map_to_rust_consts():
    src = """
import math

c = math.pi * 2.0
"""
    rust = transpile_source(src)
    assert "std::f64::consts::PI" in rust
    assert "let c: f64" in rust


def test_dep_random_maps_to_runtime_shim_with_types():
    src = """
import random

random.seed(42)
n = random.randint(1, 10)
u = random.uniform(0.0, 1.0)
r = random.random()
"""
    rust = transpile_source(src)
    assert "py_seed_int(42);" in rust
    assert "let n: i64 = py_randint(1, 10);" in rust
    assert "let u: f64 = py_uniform(0.0f64, 1.0f64);" in rust
    assert "let r: f64 = py_random();" in rust


def test_dep_random_shuffle_marks_mut_and_borrows_without_clone():
    src = """
import random

items = [3, 1, 2]
random.shuffle(items)
first = items[0]
pick = random.choice(items)
print(first, pick)
"""
    rust = transpile_source(src)
    assert "let mut items: Vec<i64>" in rust
    assert "py_shuffle(&mut items);" in rust
    assert "py_choice(&items)" in rust


def test_honesty_unsupported_constructs_raise():
    cases = [
        "try:\n    x = 1\nexcept:\n    x = 2\n",
        "gen = (x for x in [1, 2])\n",
        "print(list(map(str, [1])))\n",
        "x = [1]\nx.title()\n",
        "a = [1]\nb = a @ a\n",
        "def f():\n    yield 1\nf()\n",
        "f = lambda x: x\n",
        "x = 1\nmatch x:\n    case 1:\n        print(1)\n",
        "def outer():\n    def inner():\n        return 1\n    return inner()\nouter()\n",
        "def f():\n    global x\n    x = 1\nf()\n",
        "import math\nprint(math.gamma(10))\n",
        "xs = [3, 1]\nxs.sort(key=len)\n",
        "class A:\n    def __repr__(self):\n        return 'A'\nprint(A())\n",
        "print(isinstance(1, int))\n",
    ]
    for src in cases:
        with pytest.raises(PyRustError):
            transpile_source(src)


def test_walrus_assignment_expression():
    src = "items = [1, 2, 3]\nif (n := len(items)) > 2:\n    print(n)\n"
    rust = transpile_source(src)
    assert "let mut n: i64 = Default::default();" in rust
    assert "n = (items.len() as i64);" in rust
    assert "if (n > 2) {" in rust

    src2 = "n = 3\nwhile (n := n - 1) > 0:\n    print(n)\n"
    rust2 = transpile_source(src2)
    assert "while {" in rust2
    assert "n = (n - 1);" in rust2

    src3 = "if (m := 10):\n    print(m)\n"
    rust3 = transpile_source(src3)
    assert "m = 10;" in rust3
    assert "if (m != 0) {" in rust3


def test_boolop_or_and_yield_values_not_bools():
    src = 'user = ""\nname = user or "anon"\n'
    rust = transpile_source(src)
    assert 'let name: String = if (!user.is_empty()) { user } else { "anon".to_string() };' in rust

    src2 = "a = 0\nb = 5\nc = a or b\n"
    rust2 = transpile_source(src2)
    assert "let c: i64 = if (a != 0) { a } else { b };" in rust2

    src3 = "x = 1\ny = 2\nz = x and y\n"
    rust3 = transpile_source(src3)
    assert "let z: i64 = if (x != 0) { y } else { x };" in rust3


def test_dict_and_set_comprehensions():
    src = "sq = {k: k * k for k in range(5)}\n"
    rust = transpile_source(src)
    assert ".map(|k| (k, (k * k))).collect::<PyDict<_, _>>()" in rust
    assert "let sq: PyDict<i64, i64>" in rust

    src2 = 'uniq = {w for w in ["a", "b", "a"]}\n'
    rust2 = transpile_source(src2)
    assert ".collect::<HashSet<_>>()" in rust2


def test_fstring_format_specs():
    src = "x = 3.14159\nprint(f'{x:.2f}')\n"
    rust = transpile_source(src)
    assert 'py_format(&x, ".2f")' in rust

    src2 = "n = 42\nprint(f'{n:05d}')\n"
    rust2 = transpile_source(src2)
    assert 'py_format(&n, "05d")' in rust2

    # The full format-spec mini-language is supported (checked against CPython
    # by the differential suite); nested specs are not.
    assert 'py_format(&x, ".2%")' in transpile_source("x = 1.5\nprint(f'{x:.2%}')\n")
    with pytest.raises(PyRustError, match="nested"):
        transpile_source("x = 1.5\nw = 5\nprint(f'{x:{w}}')\n")


def test_file_io_open_read_write_and_with():
    src = 'text = open("data.txt").read()\nprint(text)\n'
    rust = transpile_source(src)
    assert 'PyFile::open_read(&"data.txt".to_string()).read()' in rust

    src2 = 'f = open("out.txt", "w")\nf.write("hi")\nf.close()\n'
    rust2 = transpile_source(src2)
    assert "let mut f: PyFile" in rust2
    assert 'PyFile::open_write(&"out.txt".to_string())' in rust2
    assert 'f.write(&"hi".to_string());' in rust2
    assert "f.close();" in rust2

    src3 = 'with open("data.txt") as f:\n    body = f.read()\nprint(body)\n'
    rust3 = transpile_source(src3)
    assert "let mut f: PyFile" in rust3
    assert "f.close();" in rust3

    src4 = "for line in open('data.txt'):\n    print(line)\n"
    rust4 = transpile_source(src4)
    assert ".lines().into_iter()" in rust4

    with pytest.raises(PyRustError, match="open\\(\\)"):
        transpile_source("f = open()\n")
    with pytest.raises(PyRustError, match="binary"):
        transpile_source('f = open("x", "rb")\n')


def test_default_args_filled_at_call_sites():
    src = 'def greet(name="world"):\n    return "hi " + name\nprint(greet())\nprint(greet("bob"))\n'
    rust = transpile_source(src)
    assert "pub fn greet(name: String) -> String" in rust
    assert 'greet("world".to_string())' in rust
    assert 'greet("bob".to_string())' in rust


def test_star_args_collect_into_vec():
    src = "def total(*nums):\n    return sum(nums)\nprint(total(1, 2, 3))\n"
    rust = transpile_source(src)
    assert "pub fn total(nums: Vec<i64>) -> i64" in rust
    assert "total(vec![1, 2, 3])" in rust

    with pytest.raises(PyRustError, match="\\*\\*kwargs"):
        transpile_source("def f(**kw):\n    return 1\n")


def test_str_format_and_percent_formatting():
    src = 's = "{} {}".format("a", 1)\nprint(s)\n'
    rust = transpile_source(src)
    assert 'format!("{} {}", "a".to_string(), 1)' in rust

    src2 = 'name = "bob"\nage = 30\ns2 = "hi %s, you are %d" % (name, age)\nprint(s2)\n'
    rust2 = transpile_source(src2)
    assert 'let s2: String = format!("hi {}, you are {}", py_str(&name), py_format(&age, "d"));' in rust2

    src3 = 's = "{name}".format(name="x")\n'
    assert 'format!("{}", "x".to_string())' in transpile_source(src3)

    with pytest.raises(PyRustError, match="keyword"):
        transpile_source('s = "{name}".format()\n')


def test_print_sep_end_and_float_display():
    src = 'print(1, 2, 3, sep=",")\n'
    rust = transpile_source(src)
    assert '.join(",")' in rust
    assert "py_write_str(" in rust

    src2 = 'print("x", end="")\n'
    rust2 = transpile_source(src2)
    assert 'py_write_str(&format!("{}{}", format!("{}", "x".to_string()), ""))' in rust2

    src3 = "print(4.0)\n"
    assert 'py_println!("{}", py_float_repr(4.0f64));' in transpile_source(src3)


def test_builtins_sweep_divmod_round_all_any():
    src = "q, r = divmod(7, 3)\nprint(q, r)\n"
    assert "let (q, r)" in transpile_source(src)

    assert "py_round_bankers" in transpile_source("print(round(2.5))\n")
    assert ".iter().copied().all(|v| v)" in transpile_source("print(all([True, True]))\n")
    assert ".iter().copied().any(|v| v)" in transpile_source("print(any([False, True]))\n")
    assert ".rev()" in transpile_source("print(list(reversed([1, 2])))\n")
    assert 'py_hex(255)' in transpile_source("print(hex(255))\n")
    assert 'py_oct(8)' in transpile_source("print(oct(8))\n")
    assert 'py_bin(5)' in transpile_source("print(bin(5))\n")
    assert 'py_ord(&"a".to_string())' in transpile_source("print(ord('a'))\n")
    assert "py_chr(65)" in transpile_source("print(chr(65))\n")
    assert "(i as i64 + 1, v)" in transpile_source("print(list(enumerate(['a'], 1)))\n")


def test_json_dumps_and_loads():
    src = 'import json\nd = {"a": 1}\ns = json.dumps(d)\nprint(s)\n'
    rust = transpile_source(src)
    assert "py_json_dumps(&d)" in rust
    assert "let s: String" in rust

    src2 = 'import json\nv = json.loads(\'{"a": 1}\')\nprint(v)\n'
    rust2 = transpile_source(src2)
    assert "py_json_loads(" in rust2
    assert "let v: PyValue" in rust2


def test_min_max_float_uses_partial_cmp():
    assert "py_min_of" in transpile_source("print(min([1.5, 0.5]))\n")


def test_methods_sweep_list_dict_str():
    assert ".pop_key(" in transpile_source('d = {"a": 1}\nd.pop("a")\n')
    assert "py_list_pop_at(" in transpile_source("xs = [1, 2, 3]\nxs.pop(0)\n")
    assert '.entry("k".to_string()).or_insert(5).clone();' in transpile_source(
        'd = {}\nd.setdefault("k", 5)\n'
    )
    # String methods go through runtime helpers with CPython's semantics
    # (Unicode whitespace/digits, character offsets, strip(chars)).
    assert "py_lstrip(&\"  x\".to_string(), None)" in transpile_source('print("  x".lstrip())\n')
    assert "py_rstrip(&\"x  \".to_string(), Some(&\" \".to_string()))" in transpile_source('print("x  ".rstrip(" "))\n')
    assert "py_str_isdigit(" in transpile_source('print("42".isdigit())\n')
    assert "py_str_find(" in transpile_source('print("abc".find("b"))\n')
    assert "py_removeprefix(" in transpile_source('print("unhappy".removeprefix("un"))\n')


def test_mixed_bool_int_containers_stay_dynamic():
    rust = transpile_source('xs = [1, True]\nprint(xs)\n')
    assert "Vec<PyValue>" in rust
    assert "PyValue::from(1)" in rust or "PyValue::from(true)" in rust


def test_tuple_print_uses_python_repr():
    rust = transpile_source('t = (1, "a")\nprint(t)\n')
    assert 'py_str(&t)' in rust


def test_split_no_arg_uses_whitespace_semantics():
    assert "py_split_ws_max(&s, -1)" in transpile_source('s = "a  b\\n"\nprint(s.split())\n')


def test_comparison_edge_cases():
    src = "x = None\nif x is None:\n    print(1)\n"
    assert "(PyValue::from(x) == PyValue::None)" in transpile_source(src)

    with pytest.raises(PyRustError, match="identity"):
        transpile_source("a = 1\nb = 2\nprint(a is b)\n")

    src2 = "print(1 < 2 < 3)\n"
    assert "&&" in transpile_source(src2)

    src3 = "t = (1, 2, 3)\nprint(2 in t)\n"
    rust3 = transpile_source(src3)
    assert "t.0" in rust3 and "t.1" in rust3 and "t.2" in rust3


def test_for_else_while_else_and_slice_assign():
    src = "found = False\nfor i in [1, 2]:\n    if i == 9:\n        found = True\n        break\nelse:\n    found = False\n"
    rust = transpile_source(src)
    assert "__py_else_" in rust
    assert "if __py_else_" in rust

    src2 = "xs = [1, 2, 3, 4]\nxs[1:3] = [9, 9]\nprint(xs)\n"
    assert "py_splice" in transpile_source(src2)

    src3 = "t = (1, 2)\nprint(1 in t)\n"
    assert "t.0" in transpile_source(src3) or "||" in transpile_source(src3)


def test_numeric_edge_cases():
    assert "((true as i64) + 1)" in transpile_source("print(True + 1)\n")
    assert "py_pow_f64(2.0, ((-1) as f64))" in transpile_source("print(2 ** -1)\n")
    assert "py_float_repr(py_mod_f64(7.5f64, 3.0f64))" in transpile_source("print(7.5 % 3.0)\n")
    assert '(if true { "True" } else { "False" })' in transpile_source("print(True)\n")
    assert "(py_sum_ints(&vec![1, 2]) + 10)" in transpile_source("print(sum([1, 2], 10))\n")


def test_dataclass_fields_generate_init():
    src = (
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class Point:\n"
        "    x: float = 0.0\n"
        "    y: float = 0.0\n\n"
        "p = Point()\n"
        "q = Point(1.0)\n"
    )
    rust = transpile_source(src)
    assert "pub x: f64," in rust
    assert "pub fn new(x: f64, y: f64) -> Self" in rust
    assert "let p: Point = Point::new(0.0f64, 0.0f64);" in rust
    assert "let q: Point = Point::new(1.0f64, 0.0f64);" in rust

    with pytest.raises(PyRustError, match="staticmethod"):
        transpile_source("class A:\n    @staticmethod\n    def f(x):\n        return x\na = A()\nprint(a.f(1))\n")
    with pytest.raises(PyRustError, match="super"):
        transpile_source("class A:\n    def m(self):\n        return super().m()\nA().m()\n")


def test_single_inheritance_flattens_fields_and_methods():
    src = (
        "class Base:\n"
        "    def __init__(self, x):\n"
        "        self.x = x\n"
        "    def get(self):\n"
        "        return self.x\n\n"
        "class Child(Base):\n"
        "    def __init__(self, x, y):\n"
        "        self.x = x\n"
        "        self.y = y\n\n"
        "c = Child(1, 2)\n"
        "print(c.get(), c.y)\n"
    )
    rust = transpile_source(src)
    assert "pub x: i64," in rust
    assert "pub fn get(&self) -> i64" in rust

    with pytest.raises(PyRustError, match="Multiple inheritance"):
        transpile_source("class C(A, B):\n    pass\n")
    with pytest.raises(PyRustError, match="base class"):
        transpile_source("class D(Unknown):\n    pass\n")


def test_dynamic_fields_and_struct_equality():
    src = "class P:\n    def __init__(self, x):\n        self.x = x\np = P(1)\np.extra = 2\nprint(p.extra)\n"
    rust = transpile_source(src, filename="main.pyr")
    assert "pub extra:" in rust
    assert "#[derive(Debug, Clone, Default, PartialEq)]" in rust


def test_stdlib_sys_os_time_shims():
    src = "import sys\nprint(sys.argv[1:])\nsys.exit(0)\n"
    rust = transpile_source(src)
    assert "std::env::args().collect::<Vec<String>>()" in rust
    assert "py_exit(((0) as i64 & 0xff) as i32)" in rust

    src2 = 'import os\nprint(os.getenv("HOME"))\nos.makedirs("d", exist_ok=True)\n'
    rust2 = transpile_source(src2)
    assert "py_os_getenv(" in rust2
    assert "py_os_mkdir(" in rust2

    src3 = "import time\nt = time.time()\ntime.sleep(0.5)\n"
    rust3 = transpile_source(src3)
    assert "py_unix_time()" in rust3
    assert "py_sleep(" in rust3

    with pytest.raises(PyRustError, match="os.path"):
        transpile_source('import os\nprint(os.path.join("a", "b"))\n')


def test_augassign_and_del_slice_and_break_else():
    rust = transpile_source("x = 7\nx %= 3\nprint(x)\n")
    assert "x = py_mod(x, 3);" in rust
    rust2 = transpile_source("x = 7\nx //= 2\nprint(x)\n")
    assert "x = py_floordiv(x, 2);" in rust2
    rust3 = transpile_source("xs = [1, 2, 3, 4]\ndel xs[1:3]\nprint(xs)\n")
    assert "py_splice(&mut xs, Some(1), Some(3), Vec::<i64>::new());" in rust3

    src = "hit = False\nfor i in [1, 2]:\n    if i == 2:\n        hit = True\n        break\nelse:\n    hit = False\nprint(hit)\n"
    rust4 = transpile_source(src)
    assert "__py_else_" in rust4
    assert "= false;" in rust4

    assert '(if (true || false) { "True" } else { "False" })' in transpile_source("print(True or False)\n")


def test_corpus_wordcount_transpiles(tmp_path):
    from pyrust.builder import emit_cargo_project

    bad = tmp_path / "bad.pyr"
    bad.write_text("import sys\nfrom collections import Counter\n", encoding="utf-8")
    with pytest.raises(PyRustError, match="collections"):
        emit_cargo_project(bad, tmp_path / "bad_out")

    wc = (
        "def top_words(path):\n"
        "    counts = {}\n"
        "    ranked = sorted(counts.items(), key=len)\n"
        "    return len(ranked)\n"
        "print(top_words('x'))\n"
    )
    # sorted() with a key function is out of scope and must fail loudly
    with pytest.raises(PyRustError, match="key="):
        transpile_source(wc)

    wc_ok = (
        "def top_words(path):\n"
        "    counts = {}\n"
        "    with open(path) as f:\n"
        "        for line in f:\n"
        "            for word in line.split():\n"
        "                w = word.lower()\n"
        "                counts[w] = counts.get(w, 0) + 1\n"
        "    return len(counts)\n"
        "print(top_words('words.txt'))\n"
    )
    rust = transpile_source(wc_ok)
    assert "PyDict" in rust
    assert "py_str_slice" not in rust


def test_stdlib_platform_shim():
    src = 'import platform\nprint(platform.system())\nprint(platform.machine())\n'
    rust = transpile_source(src)
    assert "std::env::consts::OS" in rust
    assert "std::env::consts::ARCH" in rust

    assert 'let v: String = "3.11".to_string();' in transpile_source('import platform\nv = platform.python_version()\n')
    assert '"PyRust"' in transpile_source('import platform\nprint(platform.python_implementation())\n')


def test_stdlib_statistics_shim():
    src = "import statistics\nm = statistics.mean([1, 2, 3])\nprint(m)\n"
    rust = transpile_source(src)
    assert "let m: PyValue = py_mean(" in rust

    src2 = "import statistics\nmed = statistics.median([3, 1, 2])\nprint(med)\n"
    rust2 = transpile_source(src2)
    assert "let med: PyValue = py_median(" in rust2

    # CPython computes stdev with exact fractions; native code refuses it.
    src3 = "from statistics import stdev, mode\ns = stdev([1.5, 2.5])\nprint(s)\n"
    with pytest.raises(PyRustError, match="stdev"):
        transpile_source(src3)
    assert "py_mean_floats(" in transpile_source("import statistics\nprint(statistics.mean([1.5, 2.5]))\n")

    with pytest.raises(PyRustError, match="statistics"):
        transpile_source("import statistics\nprint(statistics.geometric_mean([1, 2]))\n")


def test_stdlib_gc_shim():
    src = "import gc\nn = gc.collect()\nprint(n)\nprint(gc.isenabled())\n"
    rust = transpile_source(src)
    assert "let n: i64 = py_gc_collect()" in rust
    assert "py_gc_isenabled()" in rust

    src2 = "import gc\ngc.disable()\n"
    assert "py_gc_disable();" in transpile_source(src2)


def test_module_level_names_invisible_in_functions_is_loud():
    with pytest.raises(PyRustError, match="module-level variable"):
        transpile_source("TABLE = [1, 2]\ndef f(i):\n    return TABLE[i]\nprint(f(0))\n")


def test_mixed_containers_stay_dynamic_and_honest():
    rust = transpile_source('xs = [1, 2.5]\nprint(xs)\n')
    assert "Vec<PyValue>" in rust
    assert "PyValue::from(1)" in rust or "PyValue::Int(1)" in rust

    rust2 = transpile_source('d = {}\nd["a"] = 1\nd["b"] = "x"\nprint(d)\n')
    assert "PyDict<String, PyValue>" in rust2

    rust3 = transpile_source('xs = [1, "a"]\nprint(sum(xs))\n')
    assert "py_sum_values(&xs)" in rust3

    rust4 = transpile_source('d = {"a": 1}\nprint(d.get("zz"))\n')
    assert "unwrap_or(PyValue::None)" in rust4

    rust5 = transpile_source('print(dict(a=1, b=2))\n')
    assert "PyDict::from_vec" in rust5
    rust6 = transpile_source('d = {"a": 1}\nprint(dict(d))\n')
    assert ".clone()" in rust6

    with pytest.raises(PyRustError, match="dict"):
        transpile_source('print(dict([(1, 2)], {"a": 1}))\n')


def test_uncalled_function_defaults_to_dynamic():
    # Never called and nothing constrains `x`: must still compile.
    rust = transpile_source("def helper(x):\n    return x\nprint(42)\n")
    assert "pub fn helper(x: PyValue) -> PyValue" in rust
    rust2 = transpile_source("def helper(x):\n    return x + 1\nprint(42)\n")
    assert "pub fn helper(x: i64) -> i64" in rust2

    # Called functions still infer precisely.
    rust2 = transpile_source("def double(n):\n    return n + n\nprint(double(21))\n")
    assert "pub fn double(n: i64) -> i64" in rust2


def test_dict_subscript_assignment_uses_insert():
    src = 'd = {"a": 1}\nd["b"] = 2\nprint(d)\n'
    rust = transpile_source(src)
    assert 'd.insert("b".to_string(), 2);' in rust
    assert "py_set_index" not in rust


def test_raise_maps_to_panic_and_del_subscript():
    src = 'flag = True\nif flag:\n    raise ValueError("nope")\n'
    rust = transpile_source(src)
    assert 'py_raise(format!("ValueError: {}", "nope".to_string()));' in rust

    src2 = 'd = {"a": 1}\ndel d["a"]\n'
    assert 'd.del_key(&"a".to_string());' in transpile_source(src2)

    src3 = "lst = [1, 2, 3]\ndel lst[0]\n"
    assert "py_list_del(&mut lst, 0);" in transpile_source(src3)

    with pytest.raises(PyRustError):
        transpile_source("x = 1\ndel x\n")
    with pytest.raises(PyRustError):
        transpile_source("raise\n")


def test_ellipsis_body_and_function_level_import_skipped():
    src = "def f():\n    ...\nprint(1)\n"
    assert "fn main()" in transpile_source(src)

    src2 = "import helper\ndef f():\n    import helper\n    return helper.triple(1)\nprint(f())\n"
    assert "crate::helper::triple(1)" in transpile_source(src2, filename="main.pyr")


def test_dep_random_from_import_and_unsupported_error():
    src = """
from random import randint, uniform

n = randint(1, 6)
u = uniform(0.0, 5.0)
"""
    rust = transpile_source(src)
    assert "let n: i64 = py_randint(1, 6);" in rust
    assert "let u: f64 = py_uniform(0.0f64, 5.0f64);" in rust

    with pytest.raises(PyRustError, match="random.gauss is not supported"):
        transpile_source("import random\nx = random.gauss(0.0, 1.0)\n")


def test_dep_from_import_of_local_module_emits_qualified_call():
    src = """
from helper import triple

print(triple(4))
"""
    rust = transpile_source(src, filename="main.pyr")
    assert "crate::helper::triple(4)" in rust


def test_review_fix_9_non_copy_listcomp_filter_builtins_and_list_ops():
    src = """
words = ["hi", "", "rust"]
non_empty = [w for w in words if len(w) > 0]
pos = +5
flag = True and False
combined = [1, 2] + [3, 4]
repeated = [0] * 4
rev_nums = [x for x in range(5, 0, -1)]
s_list = sorted(combined)
"""
    rust = transpile_source(src)
    assert "|&w|" not in rust
    assert "+5" not in rust
    assert "let flag: bool =" in rust
    assert "py_list_concat(" in rust
    assert "py_list_repeat(" in rust
    assert "py_range_iter(5, 0, (-1))" in rust
    assert "py_sort(&mut __v)" in rust

