class C:
    def __init__(self):
        self.xs = []
c = C()
c.xs.append(1)
ys = c.xs
ys.append(2)
print(c.xs)
