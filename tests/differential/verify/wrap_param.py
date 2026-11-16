def wrap(x):
    return [x]

a = [1]
w = wrap(a)
w[0].append(2)
print(a, w)
