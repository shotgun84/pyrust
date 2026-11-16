from pathlib import Path

import pytest

from pyrust.builder import compute_project_hash, doctor, emit_cargo_project
from pyrust.cli import main
from pyrust.errors import PyRustError


def test_emit_cargo_project_for_single_pyr_file(tmp_path: Path):
    script = tmp_path / "hello.pyr"
    script.write_text('def sq(n):\n    return n * n\nprint(sq(12))\n', encoding="utf-8")

    out_dir = tmp_path / "rust_out"
    crate_dir = emit_cargo_project(script, out_dir)
    assert crate_dir == out_dir
    assert (out_dir / "Cargo.toml").exists()
    assert (out_dir / "src" / "main.rs").exists()
    assert (out_dir / "src" / "pyrust_rt.rs").exists()

    main_rs = (out_dir / "src" / "main.rs").read_text(encoding="utf-8")
    assert "pub fn sq(n: i64) -> i64" in main_rs
    assert "fn main()" in main_rs


def test_emit_cargo_project_for_multifile_directory(tmp_path: Path):
    pkg = tmp_path / "my_app"
    pkg.mkdir()
    (pkg / "main.pyr").write_text("import math_utils\nprint(math_utils.cube(3))\n", encoding="utf-8")
    (pkg / "math_utils.py").write_text("def cube(x):\n    return x * x * x\n", encoding="utf-8")

    out_dir = tmp_path / "crate_out"
    emit_cargo_project(pkg, out_dir)
    assert (out_dir / "src" / "main.rs").exists()
    assert (out_dir / "src" / "math_utils.rs").exists()

    main_rs = (out_dir / "src" / "main.rs").read_text(encoding="utf-8")
    sub_rs = (out_dir / "src" / "math_utils.rs").read_text(encoding="utf-8")
    assert "mod math_utils;" in main_rs
    assert "crate::math_utils::cube(3)" in main_rs
    assert "pub fn cube(x: i64) -> i64" in sub_rs


def test_compute_project_hash_changes_when_source_changes(tmp_path: Path):
    script = tmp_path / "calc.pyr"
    script.write_text("x = 1\n", encoding="utf-8")
    h1 = compute_project_hash(script)
    script.write_text("x = 2\n", encoding="utf-8")
    h2 = compute_project_hash(script)
    assert h1 != h2


def test_cli_transpile_and_doctor_commands(tmp_path: Path, capsys):
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    fib_path = examples_dir / "fibonacci.pyr"
    assert fib_path.exists()

    out_dir = tmp_path / "fib_crate"
    rc = main(["transpile", str(fib_path), "-o", str(out_dir)])
    assert rc == 0
    assert (out_dir / "Cargo.toml").exists()
    assert (out_dir / "src" / "main.rs").exists()

    doc_rc = main(["doctor"])
    assert doc_rc == 0
    captured = capsys.readouterr().out
    assert "Python" in captured
    assert "Rust" in captured

    info = doctor()
    assert info["python_ok"] is True
    assert "cargo_available" in info


def test_all_example_pyr_programs_transpile_cleanly(tmp_path: Path):
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    pyr_files = sorted(examples_dir.glob("*.pyr"))
    assert len(pyr_files) >= 3
    for idx, pyr in enumerate(pyr_files):
        out_dir = tmp_path / f"ex_{idx}"
        rc = main(["transpile", str(pyr), "-o", str(out_dir)])
        assert rc == 0
        main_rs = (out_dir / "src" / "main.rs").read_text(encoding="utf-8")
        assert "fn main()" in main_rs


def test_dep_from_import_project_infers_types_across_modules(tmp_path: Path):
    pkg = tmp_path / "from_pkg"
    pkg.mkdir()
    (pkg / "main.pyr").write_text("from helper import triple\nres = triple(4)\nprint(res)\n", encoding="utf-8")
    (pkg / "helper.py").write_text("def triple(n):\n    return n * 3\n", encoding="utf-8")
    out = tmp_path / "from_out"
    emit_cargo_project(pkg, out)
    main_rs = (out / "src" / "main.rs").read_text(encoding="utf-8")
    helper_rs = (out / "src" / "helper.rs").read_text(encoding="utf-8")
    assert "mod helper;" in main_rs
    assert "let res: i64 = crate::helper::triple(4);" in main_rs
    assert "pub fn triple(n: i64) -> i64" in helper_rs


def test_dep_from_import_class_across_modules(tmp_path: Path):
    pkg = tmp_path / "from_cls_pkg"
    pkg.mkdir()
    (pkg / "main.pyr").write_text(
        "from shapes import Point\np = Point(3.0, 4.0)\nprint(p.x)\n", encoding="utf-8"
    )
    (pkg / "shapes.py").write_text(
        "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n",
        encoding="utf-8",
    )
    out = tmp_path / "from_cls_out"
    emit_cargo_project(pkg, out)
    main_rs = (out / "src" / "main.rs").read_text(encoding="utf-8")
    assert "let p: Point = crate::shapes::Point::new(3.0f64, 4.0f64);" in main_rs


def test_honesty_unknown_third_party_import_raises(tmp_path: Path):
    script = tmp_path / "app.pyr"
    script.write_text("import requests\nprint(requests.get('http://x'))\n", encoding="utf-8")
    with pytest.raises(PyRustError, match="requests"):
        emit_cargo_project(script, tmp_path / "out")

    typing_ok = tmp_path / "tags.pyr"
    typing_ok.write_text("from typing import Optional\nx = 1\nprint(x)\n", encoding="utf-8")
    out = emit_cargo_project(typing_ok, tmp_path / "out2")
    assert (out / "src" / "main.rs").exists()


def test_dep_cross_module_defaults_and_keywords(tmp_path: Path):
    pkg = tmp_path / "kw_pkg"
    pkg.mkdir()
    (pkg / "main.pyr").write_text(
        "import helper\nprint(helper.greet())\nprint(helper.greet(name=\"bob\"))\nprint(helper.total(1, 2, 3))\n",
        encoding="utf-8",
    )
    (pkg / "helper.py").write_text(
        'def greet(name="world"):\n    return "hi " + name\n\ndef total(*nums):\n    return sum(nums)\n',
        encoding="utf-8",
    )
    out = tmp_path / "kw_out"
    emit_cargo_project(pkg, out)
    main_rs = (out / "src" / "main.rs").read_text(encoding="utf-8")
    assert 'crate::helper::greet("world".to_string())' in main_rs
    assert 'crate::helper::greet("bob".to_string())' in main_rs
    assert "crate::helper::total(vec![1, 2, 3])" in main_rs


def test_dep_transitive_chain_uses_crate_paths_and_full_types(tmp_path: Path):
    pkg = tmp_path / "chain_pkg"
    pkg.mkdir()
    (pkg / "main.pyr").write_text("import helper_a\nprint(helper_a.double(21))\n", encoding="utf-8")
    (pkg / "helper_a.py").write_text("import helper_b\n\ndef double(n):\n    return helper_b.add(n, n)\n", encoding="utf-8")
    (pkg / "helper_b.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    out = tmp_path / "chain_out"
    emit_cargo_project(pkg, out)
    main_rs = (out / "src" / "main.rs").read_text(encoding="utf-8")
    a_rs = (out / "src" / "helper_a.rs").read_text(encoding="utf-8")
    b_rs = (out / "src" / "helper_b.rs").read_text(encoding="utf-8")
    assert "mod helper_a;" in main_rs
    assert "mod helper_b;" in main_rs
    assert "crate::helper_a::double(21)" in main_rs
    assert "crate::helper_b::add(" in a_rs
    assert "pub fn double(n: i64) -> i64" in a_rs
    assert "pub fn add(a: i64, b: i64) -> i64" in b_rs


def test_dep_vendor_dir_is_transpiled(tmp_path: Path):
    pkg = tmp_path / "vend_pkg"
    (pkg / "pyrust_vendor").mkdir(parents=True)
    (pkg / "main.pyr").write_text("import vlib\nprint(vlib.triple(4))\n", encoding="utf-8")
    (pkg / "pyrust_vendor" / "vlib.py").write_text("def triple(n):\n    return n * 3\n", encoding="utf-8")
    out = tmp_path / "vend_out"
    emit_cargo_project(pkg, out)
    main_rs = (out / "src" / "main.rs").read_text(encoding="utf-8")
    assert "mod vlib;" in main_rs
    assert "crate::vlib::triple(4)" in main_rs


def test_cli_migrate_command_converts_py_files_to_pyr(tmp_path: Path):
    proj = tmp_path / "legacy_python"
    proj.mkdir()
    (proj / "app.py").write_text("x = 10\nprint(x * 2)\n", encoding="utf-8")
    (proj / "util.py").write_text("def inc(n):\n    return n + 1\n", encoding="utf-8")

    rc = main(["migrate", str(proj)])
    assert rc == 0
    assert (proj / "app.pyr").exists()
    assert (proj / "util.pyr").exists()

