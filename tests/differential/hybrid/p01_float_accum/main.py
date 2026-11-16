import collections

def total(n):
    s = 0
    for i in range(n):
        s += i * 0.5
    return s

def safe_sqrt_sum(n):
    if n <= 0:
        return 0
    acc = 0.0
    for i in range(n):
        acc += i ** 0.5
    return acc

def flag(n):
    for i in range(n):
        pass
    if n > 3:
        return True
    return 0

print(total(0), total(4))
print(safe_sqrt_sum(0), safe_sqrt_sum(3))
print(flag(5), flag(1))
print(repr(total(0)), type(total(0)).__name__)
