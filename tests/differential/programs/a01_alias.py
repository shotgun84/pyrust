grid = [[0, 0], [0, 0]]
row = grid[0]
row[1] = 7
print(grid)
d = {"k": [1]}
lst = d["k"]
lst.append(2)
print(d)
a = [1, 2]
holder = [a, a]
holder[0].append(3)
print(holder, a)
def add_item(store, item):
    store["items"].append(item)
inv = {"items": []}
add_item(inv, "x")
add_item(inv, "y")
print(inv)
