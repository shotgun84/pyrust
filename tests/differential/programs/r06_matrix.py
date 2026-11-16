def matmul(a, b):
    n = len(a); m = len(b[0]); k = len(b)
    c = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += a[i][t] * b[t][j]
            c[i][j] = s
    return c
N = 120
A = [[(i + j) % 7 / 3.0 for j in range(N)] for i in range(N)]
B = [[(i * j) % 5 / 2.0 for j in range(N)] for i in range(N)]
C = matmul(A, B)
tr = 0.0
for i in range(N):
    tr += C[i][i]
print(round(tr, 6), C[0][0], C[N-1][N-1])
