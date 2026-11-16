import collections, math

def eq_mixed(n, x):
    c = 0
    for _ in range(1):
        if n == x:
            c += 1
        if n < x:
            c += 10
        if n > x:
            c += 100
    return c

def fpow(a, b):
    r = 0.0
    for _ in range(1):
        r = a ** b
    return r

def ipow(a, b):
    r = 0
    for _ in range(1):
        r = a ** b
    return r

def m1(x):
    r = 0.0
    for _ in range(1):
        r = math.sqrt(x) + math.log(x) + math.exp(x)
    return r

def mfl(x):
    r = 0
    for _ in range(1):
        r = math.floor(x)
    return r

def mhyp(x, y):
    r = 0.0
    for _ in range(1):
        r = math.hypot(x, y) + math.atan2(x, y)
    return r

def mpow(x, y):
    r = 0.0
    for _ in range(1):
        r = math.pow(x, y)
    return r

def mlog2(x, b):
    r = 0.0
    for _ in range(1):
        r = math.log(x, b)
    return r

def sorted_first(x):
    r = 0.0
    for _ in range(1):
        r = sorted([3.0, x, 1.0, 2.0])[0] + sorted([3.0, x, 1.0, 2.0])[3]
    return r

def minlist(x):
    r = 0.0
    for _ in range(1):
        r = min([x, 1.0, 2.0])
    return r

def ord2(n):
    r = 0
    for _ in range(n):
        r = ord("ab")
    return r

def setfirst(n):
    r = 0
    for _ in range(1):
        r = list({5, n, 33, 17})[0]
    return r

def iabs(n):
    r = 0
    for _ in range(1):
        r = abs(n)
    return r

def neg(n):
    r = 0
    for _ in range(1):
        r = -n
    return r

def idiv(a, b):
    r = 0.0
    for _ in range(1):
        r = a / b
    return r

eq_mixed(1, 1.0); fpow(1.0, 1.0); ipow(1, 1); m1(1.0); mfl(1.0); mhyp(1.0, 1.0); mpow(1.0, 1.0); mlog2(1.0, 2.0)
sorted_first(1.0); minlist(1.0); setfirst(1); iabs(1); neg(1); idiv(1, 1)
def show(f, *a):
    try:
        print(f.__name__, a, repr(f(*a)))
    except Exception as e:
        print(f.__name__, a, "EXC", type(e).__name__, e)
show(eq_mixed, 9007199254740993, 9007199254740992.0)
show(eq_mixed, 9223372036854775807, 9.223372036854776e18)
show(fpow, -8.0, 1/3); show(fpow, 10.0, 400.0); show(fpow, 0.0, -1.0); show(fpow, -0.0, -1.0); show(fpow, float('inf'), 2.0); show(fpow, 2.0, float('nan')); show(fpow, -1.0, float('inf'))
show(ipow, 2, -1); show(ipow, 2, 64); show(ipow, 0, -1); show(ipow, -2, 63)
for x in [-1.0, 0.0, 1000.0, float('inf'), float('nan'), 1e-320]:
    show(m1, x)
for x in [1e300, float('inf'), float('nan'), -0.5, 9.2e18, -9.3e18]:
    show(mfl, x)
show(mhyp, 1e308, 1e308); show(mhyp, float('inf'), float('nan'))
show(mpow, 10.0, 400.0); show(mpow, -8.0, 0.5); show(mpow, 0.0, -1.0); show(mpow, float('inf'), 1.0)
show(mlog2, 8.0, 1.0); show(mlog2, 8.0, 2.0); show(mlog2, 8.0, 0.0)
show(sorted_first, float('nan')); show(minlist, float('nan'))
show(ord2, 1)
show(setfirst, 1); show(setfirst, 100)
show(iabs, -9223372036854775808); show(neg, -9223372036854775808)
show(idiv, 9007199254740993, 1); show(idiv, 7, 0); show(idiv, -9223372036854775808, -1)
