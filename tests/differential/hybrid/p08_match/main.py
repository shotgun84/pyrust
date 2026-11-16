import collections
BASE = 10

def f(n):
    t = 0
    for i in range(n):
        t += BASE
    return t

print(f(2))
match 99:
    case BASE:
        pass
print(f(2))
