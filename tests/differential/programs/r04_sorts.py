import random
random.seed(1)
def bubble(a):
    a = a[:]
    n = len(a)
    for i in range(n):
        for j in range(n - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a
def quick(a):
    if len(a) <= 1:
        return a
    p = a[len(a) // 2]
    return quick([x for x in a if x < p]) + [x for x in a if x == p] + quick([x for x in a if x > p])
def merge_sort(a):
    if len(a) <= 1:
        return a
    m = len(a) // 2
    l = merge_sort(a[:m]); r = merge_sort(a[m:])
    out = []; i = j = 0
    while i < len(l) and j < len(r):
        if l[i] <= r[j]:
            out.append(l[i]); i += 1
        else:
            out.append(r[j]); j += 1
    out.extend(l[i:]); out.extend(r[j:])
    return out
data = [(i * 7919) % 1000 for i in range(2000)]
b = bubble(data); q = quick(data); m = merge_sort(data)
print(b == q == m == sorted(data), b[:5], b[-5:])
