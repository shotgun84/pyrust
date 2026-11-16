import collections

def dict_order(n):
    d = {1: 1, 2: 2, 3: 3}
    d.pop(1)
    d[1] = n
    t = 0
    for k in d.keys():
        t = t * 10 + k
    return t

def set_iter(n):
    s = set()
    for i in range(n):
        s.add((i * 7919) % 50)
    t = 0
    for x in s:
        t = (t * 31 + x) % 1000003
    return t

def set_neg(n):
    s = {-1, -2, n}
    t = 0
    for x in s:
        t = t * 100 + (x + 10)
    return t

def comp(n):
    sq = {i % 5: i for i in range(n)}
    t = 0
    for k, v in sq.items():
        t = t * 100 + v
    return t

print(dict_order(9), set_iter(20), set_neg(3), comp(12))
