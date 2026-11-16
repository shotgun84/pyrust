LIMIT = 10


def square(x):
    return min(x * x, LIMIT * LIMIT)


def cube(x):
    return x * square(x)
