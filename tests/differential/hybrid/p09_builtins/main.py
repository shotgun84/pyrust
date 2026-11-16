import collections, builtins

def g(n):
    t = 0
    for i in range(n):
        t += abs(i - 5)
    return t

print(g(10))
builtins.abs = lambda x: 0
print(g(10))
