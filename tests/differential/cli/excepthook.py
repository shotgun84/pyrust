import collections, sys


def hook(t, v, tb):
    print("hook:", t.__name__, v)


sys.excepthook = hook
print(type(__builtins__).__name__, __loader__ is not None, "__cached__" in globals())
raise ValueError("boom")
