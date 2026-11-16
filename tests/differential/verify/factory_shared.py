class Node:
    def __init__(self, v):
        self.v = v
        self.kids = []

class Tree:
    def __init__(self):
        self.nodes = []

    def add(self, v):
        n = Node(v)
        self.nodes.append(n)
        return n

t = Tree()
n = t.add(1)
n.v = 50
print(t.nodes[0].v, n.v)
