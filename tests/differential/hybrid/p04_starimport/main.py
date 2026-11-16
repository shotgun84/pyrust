import collections
SCALE = 3

def helper(x):
    return x + 1

def compute(n):
    t = 0
    for i in range(n):
        t += i * SCALE + helper(i)
    return t

try:
    from local_settings import *
except ImportError:
    pass

print(SCALE, helper(1), compute(4))
