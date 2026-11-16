text = """The quick brown fox jumps over the lazy dog. The dog barks!
A fox, a dog, and a cat walk into a bar. "Ouch," says the cat.
It's 3 o'clock; the bar's closed."""
words = []
for line in text.split("\n"):
    for raw in line.split():
        w = raw.strip('.,!;"').lower()
        if w:
            words.append(w)
freq = {}
for w in words:
    freq[w] = freq.get(w, 0) + 1
items = list(freq.items())
items.sort()
top = []
for k, v in items:
    if v > 1:
        top.append(f"{k}={v}")
print(len(words), len(freq))
print(", ".join(top))
longest = ""
for w in words:
    if len(w) > len(longest):
        longest = w
print(longest)
caps = [w.capitalize() for w in words[:6]]
print(" ".join(caps))
