class Item:
    def __init__(self, name, price, qty):
        self.name = name
        self.price = price
        self.qty = qty
    def total(self):
        return self.price * self.qty

class Inventory:
    def __init__(self):
        self.items = []
        self.index = {}
    def add(self, item):
        self.index[item.name] = len(self.items)
        self.items.append(item)
    def restock(self, name, n):
        self.items[self.index[name]].qty += n
    def value(self):
        v = 0.0
        for it in self.items:
            v += it.total()
        return v
    def report(self):
        lines = []
        for it in self.items:
            lines.append(f"{it.name:<10}{it.qty:>5}{it.total():>10.2f}")
        return "\n".join(lines)

inv = Inventory()
inv.add(Item("widget", 2.5, 10))
inv.add(Item("gadget", 13.99, 3))
inv.add(Item("doohickey", 0.15, 1000))
inv.restock("gadget", 2)
print(inv.report())
print(f"total: {inv.value():.2f}")
