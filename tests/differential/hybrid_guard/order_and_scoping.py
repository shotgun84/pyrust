# pyrust: tier=hybrid
# Names that are not bound yet, or are bound/unbound by `except ... as`,
# `match` captures, or by the function rebinding its own global name.
import collections


def outer(n):
    t = 0
    for i in range(n):
        t += inner(i) + K
    return t


def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)


try:
    print("before inner", outer(3))
except NameError as e:
    print("before inner", "NameError", e)


def inner(x):
    return x * 2


try:
    print("before K", outer(3))
except NameError as e:
    print("before K", "NameError", e)

K = 7
print("defined", outer(3))

RATE = 2


def scaled(n):
    t = 0
    for i in range(n):
        t += RATE
    return t


print(scaled(3))
try:
    int("x")
except ValueError as RATE:
    pass
try:
    print(scaled(3))
except NameError as e:
    print("NameError", e)


def helper(x):
    return x + 1


def use_helper(n):
    t = 0
    for i in range(n):
        t += helper(i)
    return t


print(use_helper(3))
match (lambda x: x * 10):
    case helper:
        pass
print(use_helper(3))

f = fact
print(f(10))
globals()["fact"] = lambda n: 0  # the recursion inside fact now reaches the lambda
print(f(10), f(2), f(1))
