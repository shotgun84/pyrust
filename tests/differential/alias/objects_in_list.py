# pyrust: tier=native
class P:
    def __init__(self, x):
        self.x = x
        self.hist = []

    def move(self, d):
        self.x += d
        self.hist.append(d)

ps = [P(1), P(2), P(3)]
for p in ps:
    p.x += 10
for i, p in enumerate(ps):
    p.move(i)
ps[0].x = -1
ps[2].hist.append(42)
print([p.x for p in ps], [p.hist for p in ps])
