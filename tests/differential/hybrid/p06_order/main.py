import collections

def outer(n):
    t = 0
    for i in range(n):
        t += inner(i)
    return t

def uses_k(n):
    t = 0
    for i in range(n):
        t += K
    return t

try:
    print(outer(3))
except NameError as e:
    print("NameError", e)
try:
    print(uses_k(3))
except NameError as e:
    print("NameError", e)

def inner(x):
    return x * 2

K = 7
print(outer(3), uses_k(3))
