# pyrust: tier=hybrid
# Native code bakes in module constants, helpers, math attributes and builtins;
# rebinding any of them at runtime must send calls back to Python.
import builtins
import collections
import math
import sys

LIMIT = 5
SCALE = 2.5
FLAG = True


def step(x):
    return x + 1


def count(n):
    c = 0
    for i in range(n):
        if i < LIMIT:
            c += 1
    return c


def scaled(n):
    t = 0.0
    for i in range(n):
        t += SCALE if FLAG else 1.0
    return t


def circ(n):
    t = 0.0
    for i in range(n):
        t += math.pi + math.sqrt(4.0)
    return t


def run(n):
    t = 0
    for i in range(n):
        t = step(t)
    return t


def outer(n):
    # calls `run` natively, which calls `step` natively: both are dependencies
    t = 0
    for i in range(n):
        t += run(2)
    return t


def absum(n):
    t = 0
    for i in range(n):
        t += abs(i - 5)
    return t


def show(label):
    print(label, count(10), scaled(2), circ(1), run(3), outer(2), absum(10))


show("start")
globals()["LIMIT"] = 5          # same value: still exact
show("same-value")
globals()["LIMIT"] = 5.0        # equal but a float: comparisons still work, but type differs
show("float-limit")
globals()["LIMIT"] = 8
show("limit8")
setattr(sys.modules[__name__], "LIMIT", 2)
show("setattr")
globals()["SCALE"] = 2          # int now: result type changes to float + int
show("scale-int")
globals()["FLAG"] = 1           # truthy int instead of True
show("flag-int")
globals()["FLAG"] = False
show("flag-false")
math.pi = 3.0
show("math.pi")
math.sqrt = lambda x: -1.0
show("math.sqrt")
globals()["step"] = lambda x: x + 100
show("step")
globals()["abs"] = lambda x: 1   # module global shadows the builtin
show("global-abs")
del globals()["abs"]
show("del-global-abs")
builtins.abs = lambda x: 0
show("builtins.abs")
builtins.range = lambda n: [0]
show("builtins.range")
del globals()["LIMIT"]
try:
    show("deleted")
except NameError as e:
    print("NameError", e)
