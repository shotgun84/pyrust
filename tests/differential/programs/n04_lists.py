a = [5, 3, 8, 1]
a.append(10); a.insert(0, 99); a.extend([7, 7])
print(a, len(a), a.count(7), a.index(8))
a.remove(7); print(a.pop(), a.pop(0), a)
a.sort(); print(a); a.reverse(); print(a)
b = a.copy(); b.append(-1); print(a, b)
print(sorted([3, 1, 2]), sorted("hello"))
print(a[1:3], a[::-1], a[-1], a[:100], a[5:2])
c = [[0] * 3 for _ in range(3)]
c[1][1] = 5; print(c)
d = [x * x for x in range(10) if x % 2]
print(d, sum(d), min(d), max(d))
a[1:3] = [100, 200, 300]; print(a)
del a[0]; print(a)
print([1, 2] + [3], [0] * 4, 3 in [1, 2, 3], [1, [2, 3]] == [1, [2, 3]])
e = []
for i, v in enumerate(["x", "y"]):
    e.append((i, v))
print(e, list(zip([1, 2, 3], "ab")), list(reversed([1, 2, 3])))
print(list(range(10, 0, -3)), any([0, 0, 1]))
m = [3, 1, 2]
n = m
n.append(4)
print(m, n)
