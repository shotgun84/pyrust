# A local json.py wins over the stdlib (and the native json shim), cold and warm.
import json
print(json.dumps([1, 2]))
print(json.WHO)
