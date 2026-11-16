class P:
    def __init__(self, x):
        self.x = x
ps = [P(1), P(2)]
for p in ps:
    p.x += 10
print(ps[0].x, ps[1].x)
