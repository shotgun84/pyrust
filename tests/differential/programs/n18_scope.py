LIMIT = 10
counter = 0
def under(x):
    return x < LIMIT
print(under(3), under(30))
for i in range(5):
    counter += i
print(counter)
