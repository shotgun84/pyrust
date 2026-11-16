import json
data = {"name": "x", "vals": [1, 2.5, None, True], "nested": {"k": "v"}}
s = json.dumps(data)
print(s)
back = json.loads(s)
print(back["vals"], back["nested"]["k"])
print(json.dumps([1, "a\"b", "é"]))
print(json.loads('{"a": [1, {"b": null}]}'))
