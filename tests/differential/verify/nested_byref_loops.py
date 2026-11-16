class G:
    def __init__(self):
        self.items = [[0]]

gs = [G(), G()]
for g in gs:
    for it in g.items:
        it.append(1)
print([g.items for g in gs])
