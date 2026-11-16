import collections

def fdiv(a, b):
    r = 0.0
    for _ in range(1):
        r = a // b
    return r

def fmod(a, b):
    r = 0.0
    for _ in range(1):
        r = a % b
    return r

def rnd(x):
    r = 0
    for _ in range(1):
        r = round(x)
    return r

def toint(x):
    r = 0
    for _ in range(1):
        r = int(x)
    return r

def mx(a, b):
    r = 0.0
    for _ in range(1):
        r = max(a, b)
    return r

def mn(a, b):
    r = 0.0
    for _ in range(1):
        r = min(a, b)
    return r

def rp(x, n):
    r = 0.0
    for _ in range(1):
        r = round(x, n)
    return r

fdiv(1.0, 1.0); fmod(1.0, 1.0); rnd(1.0); toint(1.0); mx(1.0, 1.0); mn(1.0, 1.0); rp(1.0, 1)
for a, b in [(1.0, 0.1), (-7.5, 2.0), (-6.0, 3.0), (6.0, -3.0), (-0.0, 5.0), (0.7, 0.1), (5.0, 0.5)]:
    print("fdiv", a, b, fdiv(a, b), fmod(a, b))
for x in [2.5, 3.5, -0.5, 1e19, 1e30, float('inf'), float('-inf'), float('nan'), 9.3e18]:
    try:
        print("round", x, rnd(x))
    except Exception as e:
        print("round", x, type(e).__name__, e)
    try:
        print("int", x, toint(x))
    except Exception as e:
        print("int", x, type(e).__name__, e)
nan = float('nan')
for a, b in [(nan, 1.0), (1.0, nan), (0.0, -0.0), (-0.0, 0.0)]:
    print("max/min", a, b, mx(a, b), mn(a, b))
for x, n in [(0.125, 2), (0.375, 2), (2.675, 2), (1350.0, -2), (450.0, -2), (1250.0, -2), (2.5, 0), (-0.5, 0), (1e300, 2), (123.456, -1), (5e-324, 400)]:
    print("roundn", x, n, rp(x, n))
