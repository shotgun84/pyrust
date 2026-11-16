import collections

def dict_alias(n):
    d = {0: 0}
    e = d
    for i in range(n):
        e[i] = i
    return len(d)


def loop_var_alias(n):
    rows = [[0], [0], [0]]
    for r in rows:
        r[0] = n
    return rows[0][0] + rows[1][0] + rows[2][0]

def list_of_lists(n):
    m = []
    for i in range(n):
        m.append([i])
    t = 0
    for row in m:
        row.append(1)
    for row in m:
        t += len(row)
    return t

print(dict_alias(3), loop_var_alias(4), list_of_lists(3))
