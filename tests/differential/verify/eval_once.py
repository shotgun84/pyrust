# pyrust: tier=native
def g(x):
    print("g", x)
    return x
print(g(1) < g(2) < g(3))
print(g(5) < g(2) < g(3))
print(g(0) and g(1), g(2) or g(3))
