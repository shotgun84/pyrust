# pyrust: tier=native
adj = {0: [1, 2], 1: [3], 2: [3, 4], 3: [5], 4: [5], 5: []}
order = [0]
seen = {0: True}
for node in order:
    for nb in adj[node]:
        if nb not in seen:
            seen[nb] = True
            order.append(nb)
print(order)
l = [1, 1, 2, 3]
for x in l:
    l.remove(x)
print(l)
counts = {"a": 1, "b": 2}
for k in counts:
    counts[k] += 10
print(counts)
