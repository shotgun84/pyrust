def sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    i = 2
    while i * i <= n:
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
        i += 1
    return [i for i in range(n + 1) if is_p[i]]
ps = sieve(2000000)
print(len(ps), ps[-5:], sum(ps))
