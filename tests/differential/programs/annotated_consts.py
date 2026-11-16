# pyrust: tier=native
SCALE: float = 2.5
LIMIT: int = 7

def f(x):
    return x * SCALE + LIMIT

print(f(2))
