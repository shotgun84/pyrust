import collections, sys

def depth(n):
    if n == 0:
        return 0
    return 1 + depth(n - 1)

try:
    print(depth(10_000_000))
except RecursionError:
    print("RecursionError")
print("still alive")
