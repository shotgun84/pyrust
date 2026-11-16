import collections, sys

def depth(n):
    if n == 0:
        return 0
    return 1 + depth(n - 1)

for n in [500, 5000, 200000]:
    try:
        print(n, depth(n))
    except RecursionError:
        print(n, "RecursionError")
