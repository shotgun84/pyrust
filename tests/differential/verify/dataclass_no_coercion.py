from dataclasses import dataclass

@dataclass
class V:
    x: float

v = V(0)
w = V(1.5)
print(v, v.x, w)
