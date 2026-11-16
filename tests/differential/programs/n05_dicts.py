d = {"b": 2, "a": 1}
d["c"] = 3
print(d, len(d), d["a"], d.get("z"), d.get("z", 0))
for k in d:
    print(k, d[k])
print(list(d.keys()), list(d.values()), list(d.items()))
print(d.pop("b"), d, d.setdefault("q", 9), d)
d.update({"a": 100})
print(d, "a" in d, "zz" in d)
cnt = {}
for w in "the cat the dog the end".split():
    cnt[w] = cnt.get(w, 0) + 1
print(cnt)
print(sorted(cnt.items()))
inv = {v: k for k, v in cnt.items()}
print(inv)
nested = {"x": {"y": [1, 2]}}
nested["x"]["y"].append(3)
print(nested)
del d["a"]
print(d)
s = {1, 2, 3}
s.add(4)
print(s, 2 in s, len(s), sorted(s | {9}), sorted(s & {1, 2, 7}), sorted(s - {1}))
