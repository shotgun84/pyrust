import collections

def ulen(n):
    s = "é" * n
    t = 0
    for i in range(1):
        t = len(s)
    return t

def chr_len(n):
    t = 0
    for i in range(1):
        t = len(chr(n))
    return t

def find_pos(n):
    s = "naïve café" * n
    t = 0
    for i in range(1):
        t = s.find("café")
    return t

def upper_len(n):
    s = "straße" * n
    t = 0
    for i in range(1):
        t = len(s.upper())
    return t

def isdig(n):
    s = "١٢٣" * n
    t = 0
    for i in range(1):
        if s.isdigit():
            t = int(s) % 1000
    return t

def strip_len(n):
    s = "\x1c abc \x1c" * n
    t = 0
    for i in range(1):
        t = len(s.strip())
    return t

def split_count(n):
    s = "a\x1cb\x85c d" * n
    t = 0
    for i in range(1):
        t = len(s.split())
    return t

print(ulen(3), chr_len(233), chr_len(0x1F600), find_pos(1), upper_len(1), isdig(1), strip_len(1), split_count(1))
