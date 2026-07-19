x = [0, 16, 18, 20, 22, 24, 26, 40, 60, 80]
T = [0.0, 0.4, 0.8, 1.0, 1.0, 0.8, 0.5, 0.0, 0.0, 0.0]
H = [0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.6, 0.2]

print("Acceptable temperature or Acceptable Humidity:\n")

for i in range(len(x)):
    result = max(T[i], H[i])
    print(result, "/", x[i])