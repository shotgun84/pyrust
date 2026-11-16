# pyrust: stdin=3\n10\n-4\n7\nAda\n

n = int(input())
xs = []
for _ in range(n):
    xs.append(int(input()))
print(sum(xs), sorted(xs))
name = input("name? ")
print("hi", name)
