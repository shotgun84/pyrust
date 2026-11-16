def check(x):
    if x < 0:
        raise ValueError("negative: " + str(x))
    return x
print(check(5))
print(check(-1))
