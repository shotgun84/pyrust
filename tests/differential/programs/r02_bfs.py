grid = [
    "S.#.......",
    ".##.####..",
    "....#..#.#",
    "#.#...#...",
    "..#.#.#.#E",
]
H = len(grid); W = len(grid[0])
start = (0, 0); end = (0, 0)
for r in range(H):
    for c in range(W):
        if grid[r][c] == "S":
            start = (r, c)
        if grid[r][c] == "E":
            end = (r, c)
dist = {}
dist[start] = 0
queue = [start]
head = 0
while head < len(queue):
    r, c = queue[head]
    head += 1
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != "#" and (nr, nc) not in dist:
            dist[(nr, nc)] = dist[(r, c)] + 1
            queue.append((nr, nc))
print(dist.get(end, -1), len(dist))
