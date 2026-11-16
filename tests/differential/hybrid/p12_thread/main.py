import collections, threading

def depth(n):
    if n == 0:
        return 0
    return 1 + depth(n - 1)

def work(n):
    s = 0
    for i in range(n):
        s += i % 7
    return s

res = []
def t():
    try:
        res.append(depth(60000))
    except RecursionError:
        res.append("RecursionError")
th = threading.Thread(target=t); th.start(); th.join()
print(res)
out = [None] * 8
def w(k):
    out[k] = work(1_000_000 + k)
ts = [threading.Thread(target=w, args=(k,)) for k in range(8)]
for x in ts: x.start()
for x in ts: x.join()
print(out)
