# pyrust: tier=hybrid
# An accelerated function must look and behave like the original: tracebacks
# (no extra frames), recursion depth when the native code bails out,
# introspection, calling conventions, and argument subclasses.
import collections
import enum
import inspect
import os
import pickle
import sys
import traceback


def fact(n):
    return 1 if n <= 1 else n * fact(n - 1)


def boom(n, k=3):
    t = 0
    for i in range(n):
        t += 100 // (i - k)
    return t


def tri(n, k=2):
    t = 0
    for i in range(n):
        t += i * k
    return t


def frames(exc):
    return [(os.path.basename(f.filename), f.lineno, f.name, f.line) for f in traceback.extract_tb(exc.__traceback__)]


fact(3)
boom(1)
tri(1)
# fact(700) overflows i64 at every level: Python recursion must not use twice the frames.
print(len(str(fact(700))))
try:
    boom(10)
except ZeroDivisionError as e:
    print(frames(e))
try:
    boom(10, k=2)
except ZeroDivisionError as e:
    print(frames(e))


def deep(n):
    return fact(n)


try:
    sys.setrecursionlimit(400)
    print(len(str(deep(380))))
except RecursionError:
    print("RecursionError")
finally:
    sys.setrecursionlimit(1000)

print(inspect.signature(tri), tri.__name__, tri.__qualname__, tri.__module__)
print(tri.__code__.co_varnames[:2], tri.__code__.co_argcount, tri.__defaults__)
print(os.path.basename(tri.__code__.co_filename), tri.__code__.co_firstlineno, tri.__code__.co_name)
print(tri.__globals__ is globals(), "__pyrust_accel__" in globals(), hasattr(tri, "__wrapped__"))
print(inspect.getsource(tri).splitlines()[0], callable(tri), type(tri).__name__)
print(pickle.loads(pickle.dumps(tri))(4))
print(tri(5), tri(n=5, k=4), tri(k=3, n=5), tri(5, k=True), tri(True), tri(5, 2.5))
tri.__defaults__ = (10,)
print(tri(5))
tri.__defaults__ = (2,)


class Color(enum.IntEnum):
    RED = 3


class MyInt(int):
    def __add__(self, other):
        return MyInt(int(self) + int(other) + 1000)

    def __radd__(self, other):
        return MyInt(int(self) + int(other) + 1000)

    def __mul__(self, other):
        return MyInt(int(self) * int(other) + 1)

    __rmul__ = __mul__


print(tri(Color.RED), tri(5, MyInt(2)), tri(MyInt(5)), type(tri(5, MyInt(2))).__name__)
print(fact(Color.RED), fact(False), fact(True), fact(5.0))
print(list(map(tri, [1, 2, 3])))
