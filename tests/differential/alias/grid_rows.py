# pyrust: tier=native
grid = []
for i in range(4):
    row = []
    for j in range(4):
        row.append(i * j)
    grid.append(row)
grid[1][2] = 99
grid[3][0] += 5
for r in grid:
    print(r)
dp = [[0] * 6 for _ in range(6)]
for i in range(6):
    dp[i][0] = 1
    dp[0][i] = 1
for i in range(1, 6):
    for j in range(1, 6):
        dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
print(dp[5][5])
