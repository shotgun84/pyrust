x = 5
print(x)
x = "five"
print(x)
x = [5]
print(x)
x = None
print(x is None)
vals = [1, "a", 2.5, None, [1]]
for v in vals:
    print(v)
def ident(v):
    return v
print(ident(1), ident("s"))
