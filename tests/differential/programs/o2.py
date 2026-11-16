class Counter:
    def __init__(self):
        self.n = 0
    def inc(self):
        self.n += 1
def bump(c):
    c.inc()
c = Counter()
bump(c)
bump(c)
print(c.n)
