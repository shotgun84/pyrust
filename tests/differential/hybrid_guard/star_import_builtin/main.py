# A star import after the definitions rebinds a constant, a helper and a builtin
# the native code uses. `keep` uses none of them and must stay native.
# pyrust: tier=hybrid
import collections

RATE = 3


def helper(x):
    return x + 1


def mix(n):
    t = 0
    for i in range(n):
        t += i * RATE + helper(i) + abs(i - 2)
    return t


def keep(n):
    t = 0
    i = 0
    while i < n:
        t += i * i
        i += 1
    return t


print(mix(5), keep(5))
from overrides import *
print(mix(5), keep(5), RATE, helper(1), abs(-4))
