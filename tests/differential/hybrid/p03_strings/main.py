import collections

def digit_sum(n):
    s = 0
    t = str(n)
    for i in range(len(t)):
        s += int(t[i])
    return s

def popcount(n):
    c = 0
    for _ in range(1):
        c = bin(n).count("1")
    return c

def hexlen(n):
    c = 0
    for _ in range(1):
        c = len(hex(n))
    return c

def chr_code(n):
    c = 0
    for _ in range(1):
        c = len(chr(n))
    return c

def count_down(start, stop, step):
    c = 0
    for i in range(start, stop, step):
        c += 1
    return c

def float_digits(x):
    c = 0
    t = str(x)
    for i in range(len(t)):
        if t[i].isdigit():
            c += 1
    return c

def parse_float_len(n):
    t = 0.0
    for i in range(n):
        t += float("1_0.5")
    return t

def parse_big(n):
    t = 0
    for i in range(n):
        t += int("99999999999999999999") % 7
    return t

for f, args in [(digit_sum, (123,)), (digit_sum, (-12,)), (popcount, (5,)), (popcount, (-5,)),
                (hexlen, (255,)), (hexlen, (-1,)), (chr_code, (65,)), (chr_code, (-1,)), (chr_code, (0xD800,)),
                (chr_code, (0x110000,)),
                (count_down, (10, 0, -1)), (count_down, (0, 10, -1)), (count_down, (0, 10, 3)), (count_down, (0, 10, 0)),
                (parse_float_len, (1,)), (parse_big, (1,))]:
    try:
        print(f.__name__, args, f(*args))
    except Exception as e:
        print(f.__name__, args, "EXC", type(e).__name__, e)

float_digits(1.5)
for x in [0.1, 1e16, 1.5e16, 1.2345678901234567e+20, 1.1e-5, 3.3e-7, 2.5e-10, 1e22, 1.7976931348623157e308, 5e-324, 123456789.123, 0.3 + 0.6, 9.999999999999999e22]:
    print("float_digits", repr(x), float_digits(x), sum(c.isdigit() for c in repr(x)))
