import collections

def half(x):
    return x // 2

def g(n):
    t = 0
    for i in range(n):
        t += half(i)
    return t

def big(n):
    t = 0
    for i in range(1):
        t = int(half(n))
    return t

print(half(7.0))
print(g(5))
print(big(2**60 + 3))
