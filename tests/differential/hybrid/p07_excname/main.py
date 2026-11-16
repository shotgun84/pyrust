import collections
RATE = 2

def scaled(n):
    t = 0
    for i in range(n):
        t += RATE
    return t

print(scaled(3))
try:
    int("x")
except ValueError as RATE:
    pass
try:
    print(scaled(3))
except NameError as e:
    print("NameError", e)
