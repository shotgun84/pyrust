def reg(a, b):
    x = [0]
    a.append(x)
    b.append(x)

l1 = []
l2 = []
reg(l1, l2)
l1[0].append(1)
print(l1, l2)
