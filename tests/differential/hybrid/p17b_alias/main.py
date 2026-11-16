import collections

def alias(n):
    a = [0, 0, 0]
    b = a
    for i in range(1):
        b[0] = n
    return a[0]

def chain(n):
    a = b = []
    for i in range(n):
        a.append(i)
    return len(b)

def nested(n):
    grid = [[0] * 3] * 3
    for i in range(1):
        grid[0][0] = n
    return grid[1][0] + grid[2][0]

def rows(n):
    row = [0, 0]
    m = [row, row]
    for i in range(1):
        m[0][1] = n
    return m[1][1]

def neg_index(n):
    a = [1, 2, 3]
    r = 0
    for i in range(1):
        r = a[n]
    return r

def keyerr(n):
    d = {1: 10}
    r = 0
    for i in range(1):
        r = d[n]
    return r

def slices(n):
    a = [1, 2, 3, 4, 5, 6]
    t = 0
    for x in a[n::-2]:
        t = t * 10 + x
    return t

for f, a in [(alias, 7), (chain, 3), (nested, 5), (rows, 9), (neg_index, -1), (neg_index, -4), (neg_index, 3),
             (keyerr, 1), (keyerr, 2), (slices, 4), (slices, -1), (slices, 100), (slices, -100)]:
    try:
        print(f.__name__, a, f(a))
    except Exception as e:
        print(f.__name__, a, "EXC", type(e).__name__, e)
