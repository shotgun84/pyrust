import collections

def i_add(a, b):
    r = 0
    for _ in range(1):
        r = a + b
    return r

def i_sub(a, b):
    r = 0
    for _ in range(1):
        r = a - b
    return r

def i_mul(a, b):
    r = 0
    for _ in range(1):
        r = a * b
    return r

def i_fdv(a, b):
    r = 0
    for _ in range(1):
        r = a // b
    return r

def i_mod(a, b):
    r = 0
    for _ in range(1):
        r = a % b
    return r

def i_pw(a, b):
    r = 0
    for _ in range(1):
        r = a ** (b % 70 - 3)
    return r

def i_band(a, b):
    r = 0
    for _ in range(1):
        r = a & b
    return r

def i_bor(a, b):
    r = 0
    for _ in range(1):
        r = a | b
    return r

def i_bxor(a, b):
    r = 0
    for _ in range(1):
        r = a ^ b
    return r

def i_inv(a, b):
    r = 0
    for _ in range(1):
        r = ~a + b
    return r

def i_ab(a, b):
    r = 0
    for _ in range(1):
        r = abs(a) + b
    return r

def i_mn(a, b):
    r = 0
    for _ in range(1):
        r = min(a, b)
    return r

def i_mx(a, b):
    r = 0
    for _ in range(1):
        r = max(a, b)
    return r

def i_dm(a, b):
    r = 0
    for _ in range(1):
        r = divmod(a, b)[0] * 3 + divmod(a, b)[1]
    return r

def i_rnd(a, b):
    r = 0
    for _ in range(1):
        r = round(a) + b
    return r

def i_neg(a, b):
    r = 0
    for _ in range(1):
        r = -a - b
    return r

def i_lt(a, b):
    r = 0
    for _ in range(1):
        r = (a < b) + (a <= b) * 2 + (a == b) * 4 + (a != b) * 8
    return r

V = [0, 1, -1, 2, -2, 3, -3, 7, -7, 63, 64, -64, 2**31, 2**32+1, -(2**31), 2**62, -(2**62), 2**63-1, -(2**63), 3037000499, 3037000500, -3037000500]
i_add(1, 1)
i_sub(1, 1)
i_mul(1, 1)
i_fdv(1, 1)
i_mod(1, 1)
i_pw(1, 1)
i_band(1, 1)
i_bor(1, 1)
i_bxor(1, 1)
i_inv(1, 1)
i_ab(1, 1)
i_mn(1, 1)
i_mx(1, 1)
i_dm(1, 1)
i_rnd(1, 1)
i_neg(1, 1)
i_lt(1, 1)
for f in [i_add, i_sub, i_mul, i_fdv, i_mod, i_pw, i_band, i_bor, i_bxor, i_inv, i_ab, i_mn, i_mx, i_dm, i_rnd, i_neg, i_lt]:
    for a in V:
        for b in V:
            try:
                r = f(a, b)
                print(f.__name__, a, b, repr(r))
            except Exception as e:
                print(f.__name__, a, b, type(e).__name__, e)
