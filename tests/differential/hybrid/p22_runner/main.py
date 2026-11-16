import collections, sys

def work(n):
    t = 0
    for i in range(n):
        t += i
    return t

print(work(10))
print(type(__builtins__).__name__)
print(__loader__ is not None, __spec__, __package__)
print(sys.argv[0].endswith("prog.py"), __file__.endswith("prog.py"))
print(sorted(k for k in globals() if k.startswith("__")))
print(work.__module__, work.__globals__ is globals())
import warnings
warnings.warn("hello")
sys.stdout.flush()
raise ValueError("boom")
