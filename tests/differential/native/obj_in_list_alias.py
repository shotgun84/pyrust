class P:
    def __init__(self):
        self.v = 0
ps = [P(), P()]
p = ps[0]
p.v = 5
print(ps[0].v)
