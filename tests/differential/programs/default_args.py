# pyrust: tier=native
def helper(i, k=2):
    return i * k

def run(n):
    t = 0
    for i in range(n):
        t += helper(i)
    return t

print(run(10), helper(3, 5))
