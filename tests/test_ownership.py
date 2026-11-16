import ast

from pyrust.inference import infer_module
from pyrust.ownership import OwnershipPlan, analyze_ownership
from pyrust.parser import parse_source


def test_mutability_detected_for_reassigned_and_container_mutated_vars():
    src = """
const_num = 10
counter = 0
counter += 1
items = [1, 2]
items.append(3)
table = {"a": 1}
table["b"] = 2
"""
    parsed = parse_source(src)
    env = infer_module(parsed)
    plans = analyze_ownership(parsed, env)
    main_plan = plans["__main__"]
    assert isinstance(main_plan, OwnershipPlan)
    assert "const_num" not in main_plan.mutable_vars
    assert "counter" in main_plan.mutable_vars
    assert "items" in main_plan.mutable_vars
    assert "table" in main_plan.mutable_vars


def test_non_copy_variable_used_multiple_times_schedules_clone_on_earlier_uses():
    src = """
def consume(text):
    return text + "!"

greeting = "hello"
first = consume(greeting)
second = consume(greeting)
"""
    parsed = parse_source(src)
    env = infer_module(parsed)
    plans = analyze_ownership(parsed, env)
    main_plan = plans["__main__"]

    # Collect ast.Name load nodes for `greeting` in main_body
    load_nodes = [
        node
        for stmt in parsed.main_body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "greeting"
    ]
    assert len(load_nodes) == 2
    # The first use must be cloned so the second use remains valid in Rust
    assert id(load_nodes[0]) in main_plan.clone_expr_ids


def test_non_copy_variable_used_inside_loop_always_cloned():
    src = """
prefix = "item_"
out = []
for i in range(3):
    out.append(prefix)
"""
    parsed = parse_source(src)
    env = infer_module(parsed)
    plans = analyze_ownership(parsed, env)
    main_plan = plans["__main__"]

    load_nodes = [
        node
        for stmt in parsed.main_body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "prefix"
    ]
    assert len(load_nodes) == 1
    assert id(load_nodes[0]) in main_plan.clone_expr_ids
