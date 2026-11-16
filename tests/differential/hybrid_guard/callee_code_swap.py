# pyrust: tier=hybrid
# A function called natively has its code object replaced at runtime.
import collections


def helper(x):
    return x * 2


def other(x):
    return x + 2


def run(n):
    t = 0
    for i in range(n):
        t += helper(i)
    return t


print(run(4))
helper.__code__ = other.__code__
print(run(4))
