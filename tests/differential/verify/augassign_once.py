# pyrust: tier=native
import random
random.seed(1)
counts = {}
for k in range(6):
    counts[k] = 0
for _ in range(1000):
    counts[random.randint(0, 5)] += 1
print(counts, sum(counts.values()))
l = [0, 0, 0]
def idx():
    print("idx")
    return 1
l[idx()] += 5
print(l)
