total = 0
for i in range(100000):
    total = (total * 31 + i) % 1000000007
print(total)
h = 0
for c in "hello world":
    h = (h * 131 + ord(c)) & 0xFFFFFFFF
print(h)
