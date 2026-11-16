# pyrust: tier!=native
class Q:
    def __init__(self, items):
        self.items = items

src = [1]
q = Q(src)
q.items.append(2)
print(src, q.items)
