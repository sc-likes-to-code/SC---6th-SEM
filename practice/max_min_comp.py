R1 = [
    [0.2, 0.4, 0.7, 0.1],
    [0.5, 0.8, 0.3, 0.6],
    [0.9, 0.5, 0.4, 0.7]
]

R2 = [
    [0.6, 0.3],
    [0.8, 0.5],
    [0.4, 0.9],
    [0.7, 0.2]
]

row = 2
col = 0

max_min = 0

for y in range(4):
    value = min(R1[row][y], R2[y][col])
    max_min = max(max_min, value)
    
max_prod = 0

for y in range(4):
    value = R1[row][y] * R2[y][col]
    max_prod = max(max_prod, value)
    
print("Max-Min Composition =", max_min)
print("Max-Product Composition =", max_prod)