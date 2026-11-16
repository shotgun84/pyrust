import collections, functools

@functools.lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

def gen(n):
    for i in range(n):
        yield i

def outer(n):
    def inner(x):
        return x + n
    t = 0
    for i in range(n):
        t += inner(i)
    return t

def twice(n):
    t = 0
    for i in range(n):
        t += 1
    return t

def twice(n):
    t = 0
    for i in range(n):
        t += 2
    return t

if True:
    def cond(n):
        t = 0
        for i in range(n):
            t += 3
        return t

def apply(f, n):
    t = 0
    for i in range(n):
        t += f(i)
    return t

LST = [1, 2, 3]
def reads_list(n):
    t = 0
    for i in range(n):
        t += LST[i % 3]
    return t

def fact(n):
    return 1 if n <= 1 else n * fact(n - 1)

def ovf(n):
    t = 1
    for i in range(n):
        t = t * 1000
    return t

def kw(a, b=3):
    t = 0
    for i in range(a):
        t += b
    return t

kw(1, 2)
print(fib(80), list(gen(3)), outer(4), twice(3), cond(3), apply(abs, 3))
LST.append(10); LST[0] = 100
print(reads_list(4))
print(fact(20), fact(30), fact(5.0), fact(True))
print(ovf(5), ovf(10))
print(kw(3), kw(a=3, b=5), kw(b=2, a=4), kw(3, 2.5))
try:
    kw(3, 4, 5)
except TypeError as e:
    print("TypeError", e)
try:
    kw()
except TypeError as e:
    print("TypeError", e)
if __name__ == "__main__":
    def inmain(n):
        t = 0
        for i in range(n):
            t += i
        return t
    print(inmain(5))
