nums = [1, 2, 3, 4, 5, 6]
evens = []
for n in nums:
    if n % 2 == 0:
        evens.append(n)
nums.clear()
print(evens, nums)
matrix = [[i * 3 + j for j in range(3)] for i in range(3)]
t = [[matrix[j][i] for j in range(3)] for i in range(3)]
print(t)
s = "".join([str(i) for i in range(10)])
print(s)
