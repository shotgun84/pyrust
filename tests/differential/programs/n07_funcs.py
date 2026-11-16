def greet(name, greeting="Hello", punct="!"):
    return greeting + ", " + name + punct

def total(*nums):
    s = 0
    for n in nums:
        s += n
    return s

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def modify(lst):
    lst.append(42)

def swap(a, b):
    return b, a

print(greet("Bob"), greet("Al", punct="?"), greet(greeting="Yo", name="Z"))
print(total(), total(1, 2, 3))
print(fib(20))
x = [1]
modify(x)
print(x)
p, q = swap(1, 2)
print(p, q)
i = 0
while True:
    i += 1
    if i > 5:
        break
else:
    print("never")
for j in range(3):
    pass
else:
    print("for-else", j)
if (n := len(x)) > 1:
    print("walrus", n)
