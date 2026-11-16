# pyrust: tier=native
a = [3, 1, 2]
b = a[:]
b.sort()
c = sorted(a)
d = list(a)
d.append(9)
print(a, b, c, d)
first = a[0]
a[0] = 100
print(first, a)
