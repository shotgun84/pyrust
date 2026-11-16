def hanoi(n, a, b, c, moves):
    if n == 0:
        return
    hanoi(n - 1, a, c, b, moves)
    moves.append((a, c))
    hanoi(n - 1, b, a, c, moves)
m = []
hanoi(10, "A", "B", "C", m)
print(len(m), m[:3])
def perms(xs):
    if len(xs) <= 1:
        return [xs]
    out = []
    for i in range(len(xs)):
        for p in perms(xs[:i] + xs[i+1:]):
            out.append([xs[i]] + p)
    return out
ps = perms([1, 2, 3, 4])
print(len(ps), ps[0], ps[-1])
def ackermann(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return ackermann(m - 1, 1)
    return ackermann(m - 1, ackermann(m, n - 1))
print(ackermann(2, 3))
memo = {}
def collatz(n):
    if n == 1:
        return 0
    if n in memo:
        return memo[n]
    r = 1 + collatz(n // 2 if n % 2 == 0 else 3 * n + 1)
    memo[n] = r
    return r
best = 0
for i in range(1, 30000):
    if collatz(i) > best:
        best = collatz(i)
print(best)
