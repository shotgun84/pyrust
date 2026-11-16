# pyrust: tier=native
groups = {}
for w in ["apple", "avocado", "banana", "blueberry", "cherry"]:
    k = w[0]
    if k not in groups:
        groups[k] = []
    groups[k].append(w)
groups["a"][0] = "APPLE"
for v in groups.values():
    v.sort(reverse=True)
print(groups)
nested = {"x": {"y": 1}}
nested["x"]["y"] = 2
nested["x"]["z"] = 3
print(nested)
