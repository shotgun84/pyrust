def f():
    g = [[0]*2 for _ in range(2)]
    g[0][1] = 3
    return g
print(f())
