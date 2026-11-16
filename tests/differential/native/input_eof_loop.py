# pyrust: stdin=a\nb\n
n = 0
while True:
    s = input()
    n += 1
    if n > 100000:
        print("still looping after EOF")
        break
print(n)
