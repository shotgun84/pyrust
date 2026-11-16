import json
print(json.dumps({"b": 1, "a": 2}, sort_keys=True))
print(json.dumps([1, "é"], ensure_ascii=False))
