xs = [1, 5, 2]
big = [y for x in xs if (y := x * 2) > 3]
print(big)
