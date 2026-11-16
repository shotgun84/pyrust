def d(n):
    return 0 if n == 0 else 1 + d(n - 1)
print(d(5000))
