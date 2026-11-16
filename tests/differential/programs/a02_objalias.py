class Node:
    def __init__(self, val):
        self.val = val
        self.children = []
    def add(self, c):
        self.children.append(c)
root = Node(1)
child = Node(2)
root.add(child)
child.val = 99
child.add(Node(3))
print(root.children[0].val, len(root.children[0].children))
nodes = [root, child]
for n in nodes:
    n.val += 1
print(root.val, child.val)
