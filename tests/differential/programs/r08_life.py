W, H = 20, 10
cells = set()
for (x, y) in [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]:
    cells.add((x, y))
def step(cells):
    counts = {}
    for (x, y) in cells:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                k = ((x + dx) % W, (y + dy) % H)
                counts[k] = counts.get(k, 0) + 1
    new = set()
    for k in counts:
        n = counts[k]
        if n == 3 or (n == 2 and k in cells):
            new.add(k)
    return new
for _ in range(30):
    cells = step(cells)
for y in range(H):
    print("".join(["#" if (x, y) in cells else "." for x in range(W)]))
