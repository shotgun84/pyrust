# pyrust: tier=native
dp = [0] * 10
dp[0] = 1
for i in range(1, 10):
    dp[i] = dp[i - 1] * 2 + 1
best = [0] * 8
for w in [3, 4, 2]:
    for c in range(7, w - 1, -1):
        best[c] = max(best[c], best[c - w] + w)
print(dp, best)
