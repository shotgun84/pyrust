import collections

def f_add(a, b):
    r = 0.0
    for _ in range(1):
        r = a + b
    return r

def f_sub(a, b):
    r = 0.0
    for _ in range(1):
        r = a - b
    return r

def f_mul(a, b):
    r = 0.0
    for _ in range(1):
        r = a * b
    return r

def f_dv(a, b):
    r = 0.0
    for _ in range(1):
        r = a / b
    return r

def f_fdv(a, b):
    r = 0.0
    for _ in range(1):
        r = a // b
    return r

def f_mod(a, b):
    r = 0.0
    for _ in range(1):
        r = a % b
    return r

def f_pw(a, b):
    r = 0.0
    for _ in range(1):
        r = a ** b
    return r

def f_ab(a, b):
    r = 0.0
    for _ in range(1):
        r = abs(a) + b
    return r

def f_mn(a, b):
    r = 0.0
    for _ in range(1):
        r = min(a, b)
    return r

def f_mx(a, b):
    r = 0.0
    for _ in range(1):
        r = max(a, b)
    return r

def f_dm(a, b):
    r = 0.0
    for _ in range(1):
        r = divmod(a, b)[0] + divmod(a, b)[1]
    return r

def f_neg(a, b):
    r = 0.0
    for _ in range(1):
        r = -a
    return r

def f_lt(a, b):
    r = 0.0
    for _ in range(1):
        r = (a < b) + (a <= b) * 2 + (a == b) * 4 + (a != b) * 8
    return r

def f_rn(a, b):
    r = 0.0
    for _ in range(1):
        r = round(a, 3)
    return r

def f_ri(a, b):
    r = 0.0
    for _ in range(1):
        r = round(a) + 0.0
    return r

def f_ii(a, b):
    r = 0.0
    for _ in range(1):
        r = int(a) + 0.0
    return r

V = [0.0, -0.0, 1.0, -1.0, 0.1, -0.1, 0.5, -0.5, 1.5, 2.5, -2.5, 3.0, -3.0, 1e300, -1e300, 1e-300, 5e-324, 1e16, 2.0**53+2, float('inf'), float('-inf'), float('nan'), 0.3, 7.0, 1e308, 0.7, 2.675, 1.0005]
f_add(1.0, 1.0)
f_sub(1.0, 1.0)
f_mul(1.0, 1.0)
f_dv(1.0, 1.0)
f_fdv(1.0, 1.0)
f_mod(1.0, 1.0)
f_pw(1.0, 1.0)
f_ab(1.0, 1.0)
f_mn(1.0, 1.0)
f_mx(1.0, 1.0)
f_dm(1.0, 1.0)
f_neg(1.0, 1.0)
f_lt(1.0, 1.0)
f_rn(1.0, 1.0)
f_ri(1.0, 1.0)
f_ii(1.0, 1.0)
for f in [f_add, f_sub, f_mul, f_dv, f_fdv, f_mod, f_pw, f_ab, f_mn, f_mx, f_dm, f_neg, f_lt, f_rn, f_ri, f_ii]:
    for a in V:
        for b in V:
            try:
                r = f(a, b)
                print(f.__name__, a, b, repr(r))
            except Exception as e:
                print(f.__name__, a, b, type(e).__name__, e)
