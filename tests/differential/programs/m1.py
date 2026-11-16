c = [[0.0] * 2 for _ in range(2)]
c[0][1] = 5.0
print(c)
def f():
    d = [[0.0] * 2 for _ in range(2)]
    d[1][0] = 3.0
    return d
print(f())
A = [[(i + j) % 7 / 3.0 for j in range(3)] for i in range(2)]
print(A)
