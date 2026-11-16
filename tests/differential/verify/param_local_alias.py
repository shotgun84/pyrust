def f(x):
    y = x
    y.append(1)

def set_cell(grid, i, j, v):
    row = grid[i]
    row[j] = v

a = [0]
f(a)
g = [[0, 0], [0, 0]]
set_cell(g, 1, 0, 7)
print(a, g)
