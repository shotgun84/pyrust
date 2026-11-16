out = ""
for i in range(20000):
    out += str(i % 10)
print(len(out), out[:20], out.count("7"))
parts = []
for i in range(5):
    parts.append("row %d: %s" % (i, "*" * i))
print("\n".join(parts))
csv = "name,age,city\nann,31,paris\nbob,27,rome\ncy,45,oslo"
rows = [line.split(",") for line in csv.split("\n")]
header = rows[0]
recs = []
for r in rows[1:]:
    rec = {}
    for i in range(len(header)):
        rec[header[i]] = r[i]
    recs.append(rec)
ages = [int(r["age"]) for r in recs]
print(recs[1]["city"], sum(ages) / len(ages), max(ages))
