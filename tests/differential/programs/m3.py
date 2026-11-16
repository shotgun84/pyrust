a = [[0] * 2 for _ in range(2)]
a[0][1] = 5
print(a)
b = [[0.0, 0.0], [0.0, 0.0]]
b[1][1] = 2.5
print(b)
s = [["a", "b"], ["c", "d"]]
s[0][0] = "z"
print(s)
g = [[0] * 3 for _ in range(3)]
for i in range(3):
    g[i][i] = 1
    g[i][0] += 10
print(g)
cube = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
cube[1][0][1] = 9
print(cube)
dl = {"k": [0, 0]}
dl["k"][1] = 4
print(dl)
dd = {"a": {"b": 1}}
dd["a"]["b"] = 2
dd["a"]["c"] = 3
print(dd)
