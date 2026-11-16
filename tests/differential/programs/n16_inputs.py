# pyrust: stdin=3
10
-4
7
Ada

n = int(input())
xs = []
for _ in range(n):
    xs.append(int(input()))
print(sum(xs), sorted(xs))
name = input("name? ")
print("hi", name)
