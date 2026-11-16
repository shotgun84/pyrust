import collections, math, sys
LIMIT = 5

def count(n):
    c = 0
    for i in range(n):
        if i < LIMIT:
            c += 1
    return c

def circ(n):
    t = 0.0
    for i in range(n):
        t += math.pi
    return t

def step(x):
    return x + 1

def run(n):
    t = 0
    for i in range(n):
        t = step(t)
    return t

print(count(10), circ(1), run(3))
globals()["LIMIT"] = 8
math.pi = 3.0
globals()["step"] = lambda x: x + 100
print(count(10), circ(1), run(3))
setattr(sys.modules[__name__], "LIMIT", 2)
print(count(10))
