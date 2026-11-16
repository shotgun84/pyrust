import collections, inspect, pickle, functools

def tri(n, k=2):
    t = 0
    for i in range(n):
        t += i * k
    return t

def sentinel(n):
    r = 0
    for _ in range(1):
        r = n - 1
    return r

def isodd(n):
    r = False
    for _ in range(1):
        r = n % 2 == 1
    return r

def fails_bool(n):
    r = False
    for _ in range(1):
        r = 10 // n > 1
    return r

def boom(n):
    t = 0
    for i in range(n):
        t += 100 // (i - 3)
    return t

def nanret(x):
    r = 0.0
    for _ in range(1):
        r = x - x
    return r

tri(1); tri(1, 2); sentinel(1); isodd(1); fails_bool(1); boom(1); nanret(1.0)
print(tri(5), tri(5, 3), tri(n=5, k=4), tri(5, k=True), tri(True), tri(5, 2.5))
print(sentinel(-9223372036854775807), sentinel(-9223372036854775808))
print(isodd(3), isodd(4), isodd(True))
try:
    print(fails_bool(0))
except ZeroDivisionError as e:
    print("ZDE", e)
print(sentinel(-9223372036854775807))
try:
    boom(5)
except ZeroDivisionError as e:
    print("caught", type(e).__name__, e)
print(nanret(float('inf')), nanret(1.5))
print(inspect.signature(tri), tri.__name__, tri.__qualname__, tri.__module__)
print(tri.__code__.co_varnames[:2], tri.__code__.co_argcount, tri.__defaults__)
print(pickle.loads(pickle.dumps(tri))(4))
print(inspect.getsource(tri).splitlines()[0])
print(list(map(tri, [1, 2, 3])))
p = functools.partial(tri, k=10); print(p(3))
print(tri.__code__.co_filename.endswith("prog.py"), tri.__code__.co_firstlineno)
print(callable(tri), type(tri).__name__, hasattr(tri, "__wrapped__"))
