g = [[0] * 3 for _ in range(3)]
for i in range(3):
    g[i][i] = 1
print(g)
dp = [[0] * 5 for _ in range(5)]
for i in range(5):
    dp[i][0] = 1
    dp[0][i] = 1
for i in range(1, 5):
    for j in range(1, 5):
        dp[i][j] = dp[i-1][j] + dp[i][j-1]
print(dp[4][4])
