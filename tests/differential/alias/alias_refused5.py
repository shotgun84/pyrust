# pyrust: tier!=native
def f(x, acc=[]):
    acc.append(x)
    return acc

f(1)
print(f(2))
