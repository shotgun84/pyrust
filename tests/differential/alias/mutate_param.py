# pyrust: tier=native
def fill(xs, v):
    for i in range(len(xs)):
        xs[i] = v

def push_all(dst, n):
    for i in range(n):
        dst.append(i)

def bump_counts(counts, word):
    counts[word] = counts.get(word, 0) + 1

a = [0, 0, 0]
fill(a, 4)
b = []
push_all(b, 3)
push_all(b, 2)
c = {}
for w in "a b a c a".split():
    bump_counts(c, w)
print(a, b, c)
