with open("t.txt", "w") as f:
    f.write("a\nb\n")
with open("t.txt") as f:
    print(f.readlines())
